from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allows requests from frontend

# Load the trained pneumonia detection model
import os
from tensorflow.keras.models import load_model

# Get the absolute path of the current directory
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "pneumoniamodel.h5")

# Debugging: Print the expected file path
print(f"Looking for model at: {model_path}")

# Check if the model file actually exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ERROR: Model file not found at {model_path}")

# Load the model
model = load_model(model_path)
MODEL_PATH = "pneumonia_model.h5"  # Update with your model path
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Prepares an uploaded image for model prediction."""
    img = image.resize((224, 224))  # Adjust size to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    confidence = float(prediction[0][0])  # Assuming binary classification (0: Normal, 1: Pneumonia)
    
    result = "Pneumonia Detected" if confidence > 0.5 else "Normal"
    
    return jsonify({"result": result, "confidence": round(confidence * 100, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
