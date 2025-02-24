import cv2
import numpy as np
import pytesseract
import torch
from ultralytics import YOLO
import os
import time

# Configuration
VIDEO_PATH = "temp.mp4"
OUTPUT_PATH = "output_video.mp4"
SAVE_FRAMES = True
FRAMES_DIR = "detected_frames"
WRONG_LANE_LOG = "wrong_lane_vehicles.txt"


# Initialize models
def init_models():
    # Vehicle detection model (YOLOv8)
    vehicle_model = YOLO("models/yolov8n.pt")  # Download from https://github.com/ultralytics/assets/releases/

    # Lane detection model (You might want to use a specialized model or opencv-based solution)
    # For simplicity, we'll use a rule-based approach in this example

    return vehicle_model


def determine_lane_direction(frame, lane_info):
    """
    Determine the expected direction of travel for each lane
    This is a placeholder - you would need to implement actual lane detection
    or configure this based on your specific road layout
    """
    height, width = frame.shape[:2]

    # Example: in a two-lane road, left lane goes down, right lane goes up
    lanes = {
        "left": {"x_start": 0, "x_end": width // 2, "direction": "down"},
        "right": {"x_start": width // 2, "x_end": width, "direction": "up"}
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
            # Filter for vehicle classes (car: 2, truck: 7, bus: 5, motorcycle: 3)
            if box.cls.cpu().numpy()[0] in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                vehicles.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class": cls
                })

    return vehicles


def recognize_license_plate(vehicle_crop):
    """
    Apply OCR to recognize license plate in the vehicle crop
    """
    # Pre-process the image for better OCR results
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use pytesseract to do OCR
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    plate_text = pytesseract.image_to_string(thresh, config=config)

    # Clean up the result
    plate_text = ''.join(c for c in plate_text if c.isalnum())

    return plate_text if plate_text else "UNKNOWN"


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

        return direction != expected_direction
    else:
        # First time seeing this vehicle, add to tracking
        tracking_info[vehicle_id] = {
            "center_y": (y1 + y2) // 2,
            "license_plate": vehicle.get("license_plate", "UNKNOWN"),
            "first_seen": time.time()
        }
        return False


def get_vehicle_id(vehicle, tracking_info):
    """
    Simple vehicle tracking based on bbox overlap - you might want to use a more robust tracker
    """
    x1, y1, x2, y2 = vehicle["bbox"]
    vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Check if this vehicle overlaps with any tracked vehicle
    for vehicle_id, data in tracking_info.items():
        if time.time() - data["first_seen"] > 5:  # Remove old tracks after 5 seconds
            continue

        if "last_bbox" in data:
            prev_x1, prev_y1, prev_x2, prev_y2 = data["last_bbox"]
            # Check for overlap
            if (abs(vehicle_center[0] - (prev_x1 + prev_x2) // 2) < 50 and
                    abs(vehicle_center[1] - (prev_y1 + prev_y2) // 2) < 50):
                data["last_bbox"] = vehicle["bbox"]
                return vehicle_id

    # New vehicle, assign new ID
    new_id = max(tracking_info.keys()) + 1 if tracking_info else 1
    tracking_info[new_id] = {
        "last_bbox": vehicle["bbox"],
        "first_seen": time.time()
    }
    return new_id


def log_wrong_lane_vehicle(vehicle, frame_id, plate_text):
    """
    Log information about vehicles going in the wrong lane
    """
    with open(WRONG_LANE_LOG, 'a') as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}, Frame: {frame_id}, License: {plate_text}\n")


def process_video():
    # Initialize models
    vehicle_model = init_models()

    # Set up video capture and output
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directories if they don't exist
    if SAVE_FRAMES and not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    tracking_info = {}
    frame_id = 0

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect lane directions
        lanes = determine_lane_direction(frame, None)  # You'll need to implement actual lane detection

        # Detect vehicles
        vehicles = detect_vehicles(frame, vehicle_model)

        # Process each detected vehicle
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]

            # Extract and recognize license plate
            # In practice, you might want to use a specialized license plate detector first
            vehicle_crop = frame[y1:y2, x1:x2]
            plate_text = recognize_license_plate(vehicle_crop)
            vehicle["license_plate"] = plate_text

            # Check if vehicle is going in wrong lane
            wrong_lane = check_wrong_lane(vehicle, lanes, tracking_info)

            # Draw bounding box and information
            color = (0, 0, 255) if wrong_lane else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"Vehicle: {plate_text}"
            if wrong_lane:
                label += " (WRONG LANE)"
                log_wrong_lane_vehicle(vehicle, frame_id, plate_text)

                if SAVE_FRAMES:
                    cv2.imwrite(f"{FRAMES_DIR}/violation_{frame_id}_{plate_text}.jpg", frame)

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write output frame
        out.write(frame)

        # Progress update
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{total_frames} frames ({frame_id / total_frames * 100:.1f}%)")

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_video()