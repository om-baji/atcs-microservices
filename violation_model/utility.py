import cv2
import numpy as np
import pytesseract


class LicensePlateDetector:
    def __init__(self, config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        self.config = config
        # You could also load a specialized license plate detector model here

    def detect_and_read(self, vehicle_crop):
        """
        Detect license plate in vehicle crop and read text using OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Apply image processing to improve license plate visibility
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours that might be license plates
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and shape
        potential_plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio typical for license plates
            aspect_ratio = float(w) / h
            if 2.0 < aspect_ratio < 6.0 and w > 60 and h > 20:
                potential_plates.append((x, y, w, h))

        # Process potential plates
        results = []
        for x, y, w, h in potential_plates:
            # Extract region of interest
            roi = thresh[y:y + h, x:x + w]

            # Apply OCR
            plate_text = pytesseract.image_to_string(roi, config=self.config)

            # Clean up text
            plate_text = ''.join(c for c in plate_text if c.isalnum())

            if len(plate_text) >= 4:  # Assuming plate has at least 4 alphanumeric characters
                confidence = self._estimate_confidence(plate_text)
                results.append({
                    "text": plate_text,
                    "confidence": confidence,
                    "bbox": (x, y, w, h)
                })

        # Return best result or default
        if results:
            best_result = max(results, key=lambda x: x["confidence"])
            return best_result["text"], best_result["bbox"]

        # If no plate found, run OCR on the entire vehicle crop as fallback
        plate_text = pytesseract.image_to_string(thresh, config=self.config)
        plate_text = ''.join(c for c in plate_text if c.isalnum())
        return plate_text if plate_text else "UNKNOWN", None

    def _estimate_confidence(self, text):
        """
        Estimate confidence score based on text characteristics
        This is a simple heuristic - real systems would use more sophisticated methods
        """
        if not text:
            return 0.0

        # More characters generally means more confident detection
        length_score = min(len(text) / 10.0, 0.5)

        # Mix of letters and numbers is typical for license plates
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        diversity_score = 0.5 if (has_letters and has_numbers) else 0.0

        return length_score + diversity_score