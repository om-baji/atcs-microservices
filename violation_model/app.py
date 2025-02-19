import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
from utility import LicensePlateDetector
import config
from main import determine_lane_direction, detect_vehicles, check_wrong_lane

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Logs to a file
        logging.StreamHandler()  # Logs to console
    ]
)

app = Flask(__name__)

vehicle_model, plate_detector = None, None


def init_models():
    global vehicle_model, plate_detector
    logging.info("Loading models...")
    try:
        model_path = os.path.join(config.MODEL_PATH, config.YOLO_MODEL)
        vehicle_model = YOLO(model_path)
        plate_detector = LicensePlateDetector(config=config.OCR_CONFIG)
        logging.info("Models initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing models: {e}", exc_info=True)
        raise


init_models()


@app.route("/detect", methods=["POST"])
def detect():
    logging.info("Received request at /detect")

    if "image" not in request.files:
        logging.warning("No image file provided in request")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        logging.warning("Invalid image file received")
        return jsonify({"error": "Invalid image file"}), 400

    try:
        lanes = determine_lane_direction(image)
        vehicles = detect_vehicles(image, vehicle_model)
        tracking_info = {}
        results = []

        for vehicle in vehicles:
            plate_text, _ = plate_detector.detect_and_read(image)
            vehicle["license_plate"] = plate_text
            wrong_lane = check_wrong_lane(vehicle, lanes, tracking_info)

            results.append({
                "bbox": vehicle["bbox"],
                "confidence": vehicle["confidence"],
                "class": vehicle["class"],
                "license_plate": plate_text,
                "wrong_lane": wrong_lane
            })

        logging.info(f"Detection successful, detected {len(results)} vehicles")
        return jsonify({"detections": results})

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/detect_video", methods=["GET"])
def detect_video():
    video_path = "input_video.mp4"

    if not os.path.exists(video_path):
        logging.error("Default video file not found")
        return jsonify({"error": "Default video file not found"}), 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video file")
        return jsonify({"error": "Failed to open video file"}), 500

    frame_count = 0
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        lanes = determine_lane_direction(frame)
        vehicles = detect_vehicles(frame, vehicle_model)
        tracking_info = {}

        for vehicle in vehicles:
            plate_text, _ = plate_detector.detect_and_read(frame)
            vehicle["license_plate"] = plate_text
            wrong_lane = check_wrong_lane(vehicle, lanes, tracking_info)

            detections.append({
                "frame": frame_count,
                "bbox": vehicle["bbox"],
                "confidence": vehicle["confidence"],
                "class": vehicle["class"],
                "license_plate": plate_text,
                "wrong_lane": wrong_lane
            })

        frame_count += 1

    cap.release()
    logging.info(f"Video processing complete, analyzed {frame_count} frames")
    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(debug=True,port=5000)
