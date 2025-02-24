import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "anpr.h5"
model = tf.keras.models.load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Preprocess the image for the model
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize to model input shape
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)  # Ensure the upload directory exists
        file.save(filepath)

        # Preprocess and predict
        image = preprocess_image(filepath)
        prediction = model.predict(image)

        # Assuming the model outputs a license plate number as text
        plate_number = "".join([chr(int(x)) for x in prediction[0]])

        return jsonify({"plate_number": plate_number})

    return jsonify({"error": "Invalid file type"}), 400


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=False)