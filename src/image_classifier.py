import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os

class SuspiciousActivityClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['peeking', 'sneaking', 'stealing', 'normal']
        
    def load_model(self, model_path):
        """Load a trained YOLOv8 model from the specified path"""
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        
    def preprocess_image(self, image_path):
        """Preprocess the image for YOLOv8"""
        try:
            # YOLOv8 handles preprocessing internally
            return image_path
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
        
    def predict(self, image_path):
        """Predict the class of an image using YOLOv8"""
        try:
            if self.model is None:
                # Demo mode - return dummy results
                return {
                    'predicted_class': 'normal',
                    'confidence': 100.0,
                    'probabilities': {
                        'peeking': 0.0,
                        'sneaking': 0.0,
                        'stealing': 0.0,
                        'normal': 100.0
                    }
                }
            
            # Run YOLOv8 inference
            results = self.model(image_path)
            
            # Get the first result (since we're processing one image)
            result = results[0]
            
            # Get the detected class with highest confidence
            if len(result.boxes) > 0:
                # Get the class with highest confidence
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Get the highest confidence detection
                max_idx = np.argmax(confidences)
                predicted_class_id = class_ids[max_idx]
                confidence = float(confidences[max_idx]) * 100
                
                # Map class ID to class name
                predicted_class = self.class_names[predicted_class_id]
                
                # Calculate probabilities for all classes
                probabilities = np.zeros(len(self.class_names))
                for class_id, conf in zip(class_ids, confidences):
                    probabilities[class_id] = max(probabilities[class_id], float(conf))
                
                # Normalize probabilities
                probabilities = probabilities / np.sum(probabilities) * 100
                
            else:
                # If no objects detected, assume it's normal
                predicted_class = 'normal'
                confidence = 100.0
                probabilities = np.array([0, 0, 0, 100])
            
            # Create results dictionary
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, probabilities)
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'predicted_class': 'error',
                'confidence': 0,
                'probabilities': {
                    'peeking': 0,
                    'sneaking': 0,
                    'stealing': 0,
                    'normal': 0
                }
            } 