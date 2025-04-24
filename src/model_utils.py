from ultralytics import YOLO
from config import Config
import cv2
import os

def load_model():
    """Loads the YOLO model."""
    model_path = Config.MODEL_PATH
    print(f"Attempting to load model from: {model_path}")  # Add this line
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

def predict_image(model, image_path, conf_threshold=Config.CONF_THRESHOLD):
    """Runs model prediction on an image."""
    img = cv2.imread(image_path)
    if img is None:
      print(f"Error: Could not load image at path {image_path}")
      return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(
        source=img,
        conf=conf_threshold,
        show=False
    )
    if len(results) == 0:
        return None
    return results[0]