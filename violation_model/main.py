import cv2
import time
import os
from ultralytics import YOLO
from utility import LicensePlateDetector
import config

import torch
# print(torch.__version__)
# print(torch.version.cuda)  # Check CUDA version in PyTorch
# print(torch.backends.cudnn.version())  # Check cuDNN version
# print(torch.cuda.is_available())  # Should return True if PyTorch detects CUDA


def init_models():
    # Load vehicle detection model
    model_path = os.path.join(config.MODEL_PATH, config.YOLO_MODEL)
    vehicle_model = YOLO(model_path).to(device='cuda')

    # Initialize license plate detector
    plate_detector = LicensePlateDetector(config=config.OCR_CONFIG)

    return vehicle_model, plate_detector


def determine_lane_direction(frame):
    """
    Convert the relative lane configuration to absolute pixel values
    """
    height, width = frame.shape[:2]
    lanes = {}

    for lane_name, lane_data in config.LANE_CONFIG.items():
        x_start = int(lane_data["x_start_ratio"] * width)
        x_end = int(lane_data["x_end_ratio"] * width)

        lanes[lane_name] = {
            "x_start": x_start,
            "x_end": x_end,
            "direction": lane_data["direction"]
        }

    return lanes


def detect_vehicles(frame, model):
    """
    Detect vehicles in the frame using YOLO
    """
    results = model(frame)
    vehicles = []

    for result in results:
        boxes = result.boxes
        for box in boxes:

            if box.cls.cpu().numpy()[0] in [2, 3, 5, 7]:
                if box.conf.cpu().numpy()[0] < config.CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                vehicles.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class": cls
                })

    return vehicles


def check_wrong_lane(vehicle, lanes, tracking_info):
    """
    Check if the vehicle is going in the wrong lane by analyzing its movement direction
    """
    x1, y1, x2, y2 = vehicle["bbox"]
    vehicle_center_x = (x1 + x2) // 2

    # Find which lane the vehicle is in
    current_lane = None
    for lane_name, lane_data in lanes.items():
        if lane_data["x_start"] <= vehicle_center_x <= lane_data["x_end"]:
            current_lane = lane_name
            break

    if current_lane is None:
        return False

    # Check direction of travel by comparing with previous position
    vehicle_id = get_vehicle_id(vehicle, tracking_info)

    if vehicle_id in tracking_info:
        prev_y = tracking_info[vehicle_id]["center_y"]
        curr_y = (y1 + y2) // 2

        direction = "up" if curr_y < prev_y else "down"
        expected_direction = lanes[current_lane]["direction"]

        # Update tracking info
        tracking_info[vehicle_id]["center_y"] = curr_y
        tracking_info[vehicle_id]["last_bbox"] = vehicle["bbox"]

        return direction != expected_direction
    else:
        # First time seeing this vehicle, add to tracking
        tracking_info[vehicle_id] = {
            "center_y": (y1 + y2) // 2,
            "license_plate": vehicle.get("license_plate", "UNKNOWN"),
            "first_seen": time.time(),
            "last_bbox": vehicle["bbox"]
        }
        return False


def get_vehicle_id(vehicle, tracking_info):
    """
    Simple vehicle tracking based on bbox overlap
    """
    x1, y1, x2, y2 = vehicle["bbox"]
    vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Check if this vehicle overlaps with any tracked vehicle
    for vehicle_id, data in list(tracking_info.items()):
        if time.time() - data["first_seen"] > config.MAX_TRACKING_AGE:
            del tracking_info[vehicle_id]  # Remove old tracks
            continue

        if "last_bbox" in data:
            prev_x1, prev_y1, prev_x2, prev_y2 = data["last_bbox"]
            # Check for overlap
            if (abs(vehicle_center[0] - (prev_x1 + prev_x2) // 2) < config.OVERLAP_THRESHOLD and
                    abs(vehicle_center[1] - (prev_y1 + prev_y2) // 2) < config.OVERLAP_THRESHOLD):
                return vehicle_id

    # New vehicle, assign new ID
    new_id = max(tracking_info.keys()) + 1 if tracking_info else 1
    return new_id


def log_wrong_lane_vehicle(vehicle, frame_id, plate_text):
    """
    Log information about vehicles going in the wrong lane
    """
    with open(config.WRONG_LANE_LOG, 'a') as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}, Frame: {frame_id}, License: {plate_text}\n")


def process_video():
    # Initialize models
    vehicle_model, plate_detector = init_models()

    # Set up video capture and output
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directories if they don't exist
    if config.SAVE_FRAMES and not os.path.exists(config.FRAMES_DIR):
        os.makedirs(config.FRAMES_DIR)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (width, height))

    tracking_info = {}
    frame_id = 0

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect lane directions
        lanes = determine_lane_direction(frame)

        # Detect vehicles
        vehicles = detect_vehicles(frame, vehicle_model)

        # Draw lane boundaries if enabled
        if config.DRAW_LANES:
            for lane_name, lane_data in lanes.items():
                x_start, x_end = lane_data["x_start"], lane_data["x_end"]
                direction_text = lane_data["direction"]

                # Draw lane boundary lines
                cv2.line(frame, (x_start, 0), (x_start, height), (255, 255, 0), 2)
                cv2.line(frame, (x_end, 0), (x_end, height), (255, 255, 0), 2)

                # Draw lane direction text
                lane_center = (x_start + x_end) // 2
                cv2.putText(frame, f"Lane: {direction_text}",
                            (lane_center - 60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Process each detected vehicle
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]

            # Extract vehicle crop for license plate detection
            vehicle_crop = frame[y1:y2, x1:x2]

            # Detect license plate
            plate_text, plate_bbox = plate_detector.detect_and_read(vehicle_crop)
            vehicle["license_plate"] = plate_text

            # Check if vehicle is going in wrong lane
            wrong_lane = check_wrong_lane(vehicle, lanes, tracking_info)

            # Draw bounding box and information
            color = (0, 0, 255) if wrong_lane else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw license plate information
            label = f"Vehicle: {plate_text}"
            if wrong_lane:
                label += " (WRONG LANE)"
                log_wrong_lane_vehicle(vehicle, frame_id, plate_text)

                if config.SAVE_FRAMES and config.HIGHLIGHT_VIOLATIONS:
                    violation_path = f"{config.FRAMES_DIR}/violation_{frame_id}_{plate_text}.jpg"
                    cv2.imwrite(violation_path, frame)

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # If license plate bounding box detected, draw it on the vehicle
            if plate_bbox:
                px, py, pw, ph = plate_bbox
                cv2.rectangle(frame[y1:y2, x1:x2], (px, py), (px + pw, py + ph), (255, 0, 0), 2)

        # Write output frame
        out.write(frame)

        # Progress update
        frame_id += 1
        if frame_id % 100 == 0:
            progress = frame_id / total_frames * 100
            print(f"Processed {frame_id}/{total_frames} frames ({progress:.1f}%)")

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {config.OUTPUT_PATH}")


if __name__ == "__main__":
    # Create model directory if it doesn't exist
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    # Check if YOLO model exists, if not print instruction to download
    model_file = os.path.join(config.MODEL_PATH, config.YOLO_MODEL)
    if not os.path.exists(model_file):
        print(f"YOLO model not found at {model_file}")
        print(f"Please download it using:")
        print(
            f"wget https://github.com/ultralytics/assets/releases/download/v0.0.0/{config.YOLO_MODEL} -P {config.MODEL_PATH}")
        exit(1)

    process_video()