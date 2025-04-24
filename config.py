import os

class Config:
    DATASET_PATH = 'data/action_detection_dataset'
    TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
    TEST_DIR = os.path.join(DATASET_PATH, 'test')
    CLASSES = ['Normal', 'Peaking', 'Sneaking', 'Stealing']
    CONF_THRESHOLD = 0.25
    BATCH_SIZE = 1
    IMG_SIZE = 640
    MODEL_PATH = 'models/yolov8n.pt'