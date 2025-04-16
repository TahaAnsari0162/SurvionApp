import matplotlib.pyplot as plt
import cv2
from config import Config
from src.model_utils import predict_image
import numpy as np
from utils.visualization import display_image_grid, create_bar_chart

def classify_action(detections):
    """Classifies the action based on object detections."""
    detected_objects = [d[0] for d in detections]

    action_scores = {
        'Stealing': 0.0,
        'Sneaking': 0.0,
        'Peaking': 0.0,
        'Normal': 0.0
    }
    if 'person' in detected_objects:
        if any(obj in detected_objects for obj in ['backpack', 'handbag', 'suitcase']):
            action_scores['Stealing'] += 0.4
        if 'refrigerator' in detected_objects:
            action_scores['Stealing'] += 0.3
        if [conf for obj, conf in detections if obj == 'person'][0] < 0.6:
            action_scores['Sneaking'] += 0.5
        if len(detected_objects) <= 2:
            action_scores['Peaking'] += 0.5

    if not any(score > 0.3 for score in action_scores.values()):
        action_scores['Normal'] = 0.4

    return action_scores

def detect_action(model, image_path, conf_threshold=Config.CONF_THRESHOLD):
    print("Detecting action for:", image_path)
    result = predict_image(model, image_path, conf_threshold)
    print("After predict_image call.")
    if result is None:
        print(f"No objects detected in {image_path}")
        return None

    detections = [
        (model.names[int(box.cls[0])], float(box.conf[0]))
        for box in result.boxes
    ]

    action_scores = classify_action(detections)
    im_array = result.plot()
    image = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    actions = list(action_scores.keys())
    scores = list(action_scores.values())
    bar_chart = create_bar_chart(actions, scores)


    print("\nDetected Objects:")
    for obj, conf in detections:
        print(f"- {obj}: {conf:.2%}")

    print("\nAction Analysis:")
    predicted_action = max(action_scores.items(), key=lambda x: x[1])
    print(f"Predicted Action: {predicted_action[0]} ({predicted_action[1]:.2%} confidence)")
    print("\nAll Action Scores:")
    for action, score in action_scores.items():
        print(f"- {action}: {score:.2%}")

    return image, detections, predicted_action, action_scores, bar_chart