import os
import sys
import cv2
# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for
from src.model_utils import load_model, predict_image
from src.action_detector import detect_action
from config import Config
import uuid
import numpy as np
from utils.visualization import display_image_grid, create_pie_chart, create_barplot

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web_app/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = load_model()
if model is None:
    exit()
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            print("No file part in the request")  # Debugging line
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            print("No selected file")  # Debugging line
            return render_template('index.html', error='No selected file')
        if file:
            try:
                image_name = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
                file.save(image_path)
                action_data = detect_action(model, image_path)
                return render_template('results.html', image_path=image_path, action_data=action_data)
            except Exception as e:
                print(f"Error during image processing: {e}")  # Debugging line
                return render_template('index.html', error=str(e)) # Display error message
    except Exception as e:
        print(f"Error in upload_file function: {e}")  # Debugging line
        return render_template('index.html', error=str(e))
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))
@app.route('/data_explorer')
def data_explorer():
    class_counts = {}
    total_train_images = 0
    total_test_images = 0
    
    for class_name in Config.CLASSES:
        train_count = len(os.listdir(os.path.join(Config.TRAIN_DIR, class_name)))
        test_count = len(os.listdir(os.path.join(Config.TEST_DIR, class_name)))
        class_counts[class_name] = {'train': train_count, 'test': test_count}
        total_train_images += train_count
        total_test_images += test_count
    total_images = total_train_images + total_test_images
    
    bar_x = Config.CLASSES
    bar_y_train = [counts['train'] for counts in class_counts.values()]
    bar_y_test = [counts['test'] for counts in class_counts.values()]
    bar_chart = create_barplot(bar_x, bar_y_train, "Dataset Distribution", "Classes", "Number of Images")
    pie_labels = [f"{class_name} (Train)" for class_name in Config.CLASSES] + \
                 [f"{class_name} (Test)" for class_name in Config.CLASSES]
    pie_sizes = [counts['train'] for counts in class_counts.values()] + \
                 [counts['test'] for counts in class_counts.values()]
    pie_chart = create_pie_chart(pie_labels, pie_sizes)
    
    sample_images = []
    sample_titles = []
    for class_name in Config.CLASSES:
        class_path = os.path.join(Config.TRAIN_DIR, class_name)
        images = os.listdir(class_path)
        for sample_idx in range(3):
            img_path = os.path.join(class_path, np.random.choice(images))
            sample_images.append(img_path)
            sample_titles.append(f'{class_name}\nSample {sample_idx + 1}')
    image_grid_base64 = display_image_grid(sample_images, sample_titles, rows=len(Config.CLASSES))

    return render_template('data_explorer.html', total_images=total_images, bar_chart=bar_chart, pie_chart=pie_chart, sample_images=image_grid_base64)
@app.route('/batch_processing')
def batch_processing():
    image_paths = []
    for class_name in Config.CLASSES:
        class_path = os.path.join(Config.TEST_DIR, class_name)
        if os.path.exists(class_path):
            class_images = os.listdir(class_path)
            image_paths.extend(
                [os.path.join(class_path, img) for img in
                 class_images[:Config.BATCH_SIZE]]
            )
    if not image_paths:
        return render_template('batch_processing.html', error="No images found for batch processing.")
    images_list = []
    titles_list = []
    for image_path in image_paths:
        result = predict_image(model, image_path, conf_threshold=Config.CONF_THRESHOLD)
        if result is not None:
            im_array = result.plot()
            images_list.append(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
            titles_list.append(os.path.basename(image_path))
    image_grid_base64 = display_image_grid(images_list, titles_list, rows=2, cols=4, figsize=(20,5))
    return render_template('batch_processing.html', images=image_grid_base64)
@app.route('/analysis')
def analysis():
    results = [
        "1 person, 1 toilet",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person, 2 backpacks",
        "1 person, 1 parking meter, 1 backpack",
        "1 person, 1 backpack, 1 refrigerator",
        "1 person, 1 parking meter, 1 backpack",
        "1 person, 2 backpacks",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person, 1 backpack",
        "1 person",
        "1 person",
        "1 person",
        "1 person",
        "1 person",
        "1 person, 1 backpack",
        "1 person",
        "1 person",
        "1 person, 1 handbag, 1 refrigerator",
        "1 person, 1 backpack, 1 refrigerator",
        "1 person, 2 backpacks",
        "2 persons, 1 backpack, 1 suitcase, 1 refrigerator",
        "1 person, 1 backpack",
        "1 person, 1 backpack, 1 refrigerator",
        "1 person, 1 backpack",
        "2 persons, 1 backpack, 1 suitcase, 1 refrigerator"
    ]
    detections = {
        'person': 0,
        'backpack': 0,
        'handbag': 0,
        'suitcase': 0,
        'refrigerator': 0,
        'multiple_persons': 0
    }

    for line in results:
        if 'persons' in line:
            detections['multiple_persons'] += 1
        if 'person' in line:
            detections['person'] += 1
        if 'backpack' in line:
            detections['backpack'] += 1
        if 'handbag' in line:
            detections['handbag'] += 1
        if 'suitcase' in line or 'suitcases' in line:
            detections['suitcase'] += 1
        if 'refrigerator' in line:
            detections['refrigerator'] += 1

    total_images = len(results)

    patterns = {
        "Person with backpack": sum(1 for line in results if 'person' in line and 'backpack' in line),
        "Person with handbag": sum(1 for line in results if 'person' in line and 'handbag' in line),
        "Scenes with refrigerator": sum(1 for line in results if 'refrigerator' in line)
    }

    bar_chart = create_barplot(detections.keys(), detections.values(), 'Common Objects Detected in Stealing Scenes', 'Object', 'Frequency')

    scene_types = {
        'shop_theft': 0,
        'baggage_theft': 0,
        'other_theft': 0
    }

    for line in results:
        if 'refrigerator' in line:
            scene_types['shop_theft'] += 1
        elif any(item in line for item in ['backpack', 'handbag', 'suitcase', 'suitcases']):
            scene_types['baggage_theft'] += 1
        else:
            scene_types['other_theft'] += 1
    pie_chart = create_pie_chart(scene_types.keys(), scene_types.values())

    return render_template('analysis.html', detections=detections,
                                 total_images=total_images, patterns=patterns, bar_chart=bar_chart, pie_chart=pie_chart, scene_types = scene_types)
@app.route('/live_test')
def live_test():
    test_paths = {
        'Normal': Config.TEST_DIR + '/Normal/Normal_10.jpg',
        'Peaking': Config.TEST_DIR + '/Peaking/Peaking_10.jpg',
        'Sneaking': Config.TEST_DIR + '/Sneaking/Sneaking_10.jpg',
        'Stealing': Config.TEST_DIR + '/Stealing/Stealing_10.jpg'
    }
    test_results = {}
    for action, image_path in test_paths.items():
        try:
            action_data = detect_action(model, image_path)
            if action_data:
                test_results[action] = {
                    'image':action_data[0],
                    'detections':action_data[1],
                    'predicted_action': action_data[2],
                    'action_scores': action_data[3],
                    'bar_chart': action_data[4]
                }
        except Exception as e:
            test_results[action] = {'error':str(e)}
    return render_template('live_test.html', test_results=test_results)

if __name__ == '__main__':
    app.run(debug=True)