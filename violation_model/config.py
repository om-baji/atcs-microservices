# Configuration settings for the vehicle detection and OCR system

# Paths
VIDEO_PATH = "anp.mp4"
OUTPUT_PATH = "anp_output_video.mp4"
MODEL_PATH = "models/"
YOLO_MODEL = "yolov8n.pt"  # Use yolov8m.pt or yolov8l.pt for better accuracy
OCR_CONFIG = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Processing options
SAVE_FRAMES = True
FRAMES_DIR = "detected_frames"
WRONG_LANE_LOG = "wrong_lane_vehicles.txt"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for vehicle detection

# Lane configuration - customize based on your specific video
# This example assumes a two-lane road with left lane going down, right lane going up
LANE_CONFIG = {
    "left": {"x_start_ratio": 0.0, "x_end_ratio": 0.5, "direction": "down"},
    "right": {"x_start_ratio": 0.5, "x_end_ratio": 1.0, "direction": "up"}
}

# Vehicle tracking parameters
MAX_TRACKING_AGE = 5  # seconds
OVERLAP_THRESHOLD = 50  # pixels

# Visualization options
DRAW_LANES = True
HIGHLIGHT_VIOLATIONS = True