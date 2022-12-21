import os
from pathlib import Path
from threading import Lock, Thread
from sl_ai.dataset import GestureDataset
from sl_ai.gesture_classifier import GestureClassifier

GESTURE_PATH = Path('sl_ai/gestures_dataset.csv')
UPLOADED_GESTURE_PATH = Path('sl_ai/uploaded_gestures_dataset.csv')
MODEL_FILE_PATH = Path('sl_ai/model.h5')

class SingletonMeta(type):
    _instances = {}

    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Classifier(metaclass=SingletonMeta):
    gesture_dataset: GestureDataset = GestureDataset()
    gesture_classifier: GestureClassifier

    def __init__(self):
        print("Created a Classifier instance.")
        self.gesture_dataset.load_from_csv(GESTURE_PATH)

        if UPLOADED_GESTURE_PATH.exists():
            print("Detected user uploaded content. Adding it to the dataset.")
            uploaded_dataset: GestureDataset = GestureDataset()
            uploaded_dataset.load_from_csv(UPLOADED_GESTURE_PATH)
            self.gesture_dataset.append_dataset(uploaded_dataset)
            self.gesture_dataset.summary()

        self.gesture_classifier: GestureClassifier = GestureClassifier(gesture_dataset=self.gesture_dataset)


        if MODEL_FILE_PATH.exists():
            self.gesture_classifier.load_saved_model(model_path=MODEL_FILE_PATH)
        else:
            print('No model file found. Training a new model.')
            self.gesture_classifier.train(save_path=MODEL_FILE_PATH)

        self.gesture_classifier.summary()

# classifier = None