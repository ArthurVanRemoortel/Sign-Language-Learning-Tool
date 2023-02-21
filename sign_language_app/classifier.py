import os
import threading
from pathlib import Path
from threading import Lock, Thread

from learning_site import settings
from sl_ai.dataset import GestureDataset
from sl_ai.gesture_classifier import GestureClassifier
from sl_ai.utils import clean_listdir

# MAIN_GESTURE_DATASET_PATH = Path("sl_ai/gestures_dataset.csv")
# DATASET_LOCATION = Path("ai_data/vgt-all")
# MODEL_FILE_PATH = Path("sl_ai/model.h5")


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
        # self.load_dataset()
        self.gesture_classifier: GestureClassifier = GestureClassifier(
            gesture_dataset=self.gesture_dataset
        )

    def load_dataset(self):
        handedness_data = {}
        for gesture_folder in clean_listdir(settings.VGT_GESTURES_ROOT):
            *gestures_name_parts, handedness_string = gesture_folder.split("_")
            gesture_name = "_".join(gestures_name_parts)
            handedness_data[gesture_name] = (
                handedness_string[0] == "1",
                handedness_string[1] == "1",
            )
        self.gesture_dataset.scan_videos(dataset_location=settings.VGT_GESTURES_ROOT, handedness_data=handedness_data)
        self.gesture_dataset.load_gestures_from_csv()
        print("Searching for user uploaded datasets...")
        for user_folder in clean_listdir(settings.UPLOADED_GESTURES_ROOT):
            for gesture_folder in clean_listdir(settings.UPLOADED_GESTURES_ROOT / user_folder):
                csv_files = list(
                    filter(
                        lambda file: file.endswith(".csv"),
                        clean_listdir(
                            settings.UPLOADED_GESTURES_ROOT / user_folder / gesture_folder
                        ),
                    )
                )
                if not csv_files:
                    print(
                        "Warning: Found a uploaded gesture folder without a dataset.csv file."
                    )
                    continue
                dataset_file = csv_files[0]
                print(
                    f"Loading dataset from {settings.UPLOADED_GESTURES_ROOT / user_folder / gesture_folder / dataset_file}"
                )
                *gestures_words, handedness_string = gesture_folder.split("_")
                gesture_name = "_".join(gestures_words)
                gesture_dataset = GestureDataset(single_gesture=True)
                gesture_dataset.scan_videos(
                    dataset_location=settings.UPLOADED_GESTURES_ROOT
                    / user_folder
                    / gesture_folder,
                    handedness_data={
                        gesture_name: (
                            handedness_string[0] == "1",
                            handedness_string[1] == "1",
                        )
                    },
                )
                gesture_dataset.load_from_csv(
                    csv_path=settings.UPLOADED_GESTURES_ROOT
                    / user_folder
                    / gesture_folder
                    / dataset_file
                )
                self.gesture_dataset.append_dataset(gesture_dataset)

        if settings.SAVED_MODEL_PATH.exists():
            self.gesture_classifier.load_saved_model(model_path=settings.SAVED_MODEL_PATH)
        else:
            print("No model file found. Training a new model.")
            self.gesture_classifier.train(save_path=settings.SAVED_MODEL_PATH)
        self.gesture_classifier.summary()

