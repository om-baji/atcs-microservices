import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import joblib
import logging
from flask import Flask, request, jsonify
from dataclasses import dataclass
import threading
import time
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import queue
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS = Counter('traffic_predictions_total', 'Total number of predictions made')
VEHICLE_COUNT = Gauge('vehicle_count', 'Current vehicle count per direction', ['direction'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time taken for prediction')
MODEL_ERRORS = Counter('model_errors_total', 'Total number of model errors')


@dataclass
class TrafficData:
    """Class for holding traffic data for an intersection"""
    timestamp: datetime
    north_count: int
    south_count: int
    east_count: int
    west_count: int
    waiting_time: Dict[str, float]
    peak_hour: bool
    weather_condition: str


class IntersectionController:
    """Main controller for a traffic intersection"""

    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.min_green_time = 10  # minimum green light duration
        self.max_green_time = 120  # maximum green light duration
        self.yellow_time = 3
        self.all_red_time = 2
        self.current_phase = None
        self.last_phase_change = datetime.now()
        self.vehicle_queues = {
            'north': queue.Queue(),
            'south': queue.Queue(),
            'east': queue.Queue(),
            'west': queue.Queue()
        }

    def update_vehicle_counts(self, counts: Dict[str, int]):
        """Update vehicle counts from sensors"""
        for direction, count in counts.items():
            VEHICLE_COUNT.labels(direction=direction).set(count)
            self.vehicle_queues[direction].put(count)


class TrafficPredictionModel:
    """ML model for predicting optimal signal timings"""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'north_count', 'south_count', 'east_count', 'west_count',
            'waiting_time_north', 'waiting_time_south', 'waiting_time_east', 'waiting_time_west',
            'is_peak_hour', 'hour_of_day', 'day_of_week'
        ]

    def preprocess_data(self, data: List[TrafficData]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess traffic data for model training"""
        df = pd.DataFrame([
            {
                'north_count': d.north_count,
                'south_count': d.south_count,
                'east_count': d.east_count,
                'west_count': d.west_count,
                'waiting_time_north': d.waiting_time['north'],
                'waiting_time_south': d.waiting_time['south'],
                'waiting_time_east': d.waiting_time['east'],
                'waiting_time_west': d.waiting_time['west'],
                'is_peak_hour': d.peak_hour,
                'hour_of_day': d.timestamp.hour,
                'day_of_week': d.timestamp.weekday(),
            }
            for d in data
        ])

        # Calculate target (optimal green time) based on vehicle counts and waiting times
        y = df.apply(self._calculate_optimal_time, axis=1)
        X = df[self.feature_columns]

        return X, y

    def _calculate_optimal_time(self, row):
        """Calculate optimal green time based on traffic conditions"""
        base_time = 30
        vehicle_factor = 2  # seconds per vehicle
        waiting_factor = 1.5  # penalty for waiting time

        total_vehicles = (row['north_count'] + row['south_count'] +
                          row['east_count'] + row['west_count'])
        max_waiting_time = max(
            row['waiting_time_north'], row['waiting_time_south'],
            row['waiting_time_east'], row['waiting_time_west']
        )

        optimal_time = (
                base_time +
                vehicle_factor * total_vehicles +
                waiting_factor * max_waiting_time
        )

        return np.clip(optimal_time, 10, 120)  # Constrain between 10 and 120 seconds

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, traffic_data: TrafficData) -> float:
        """Make a prediction for optimal signal timing"""
        try:
            with PREDICTION_LATENCY.time():
                features = pd.DataFrame([{
                    'north_count': traffic_data.north_count,
                    'south_count': traffic_data.south_count,
                    'east_count': traffic_data.east_count,
                    'west_count': traffic_data.west_count,
                    'waiting_time_north': traffic_data.waiting_time['north'],
                    'waiting_time_south': traffic_data.waiting_time['south'],
                    'waiting_time_east': traffic_data.waiting_time['east'],
                    'waiting_time_west': traffic_data.waiting_time['west'],
                    'is_peak_hour': traffic_data.peak_hour,
                    'hour_of_day': traffic_data.timestamp.hour,
                    'day_of_week': traffic_data.timestamp.weekday()
                }])

                features_scaled = self.scaler.transform(features[self.feature_columns])
                prediction = self.model.predict(features_scaled)[0]
                PREDICTIONS.inc()

                return prediction
        except Exception as e:
            MODEL_ERRORS.inc()
            logger.error(f"Prediction error: {str(e)}")
            return 30.0  # fallback to default timing

    def save(self, path: str):
        """Save model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)

    def load(self, path: str):
        """Load model and scaler"""
        saved_objects = joblib.load(path)
        self.model = saved_objects['model']
        self.scaler = saved_objects['scaler']

