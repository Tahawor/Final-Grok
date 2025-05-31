from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model once
try:
    model = YOLO("train.pt")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image directly without saving
        image = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run prediction
        results = model(image, verbose=False)  # Set verbose=False to reduce logs

        # Extract class names from results
        names = model.names  # mapping from class id to name
        detected_classes = []

        for r in results:
            if r.boxes is not None:
                for c in r.boxes.cls:
                    class_id = int(c.item())
                    class_name = names[class_id]
                    detected_classes.append(class_name)

        # Remove duplicates
        detected_classes = list(set(detected_classes))

        if not detected_classes:
            detected_classes = ["no object detected"]

        logger.info(f"Detected classes: {detected_classes}")
        return jsonify({"detected_classes": detected_classes})
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)