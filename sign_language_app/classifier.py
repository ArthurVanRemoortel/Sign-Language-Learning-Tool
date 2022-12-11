import os
from pathlib import Path

from sl_ai.dataset import GestureDataset
from sl_ai.gesture_classifier import GestureClassifier

gesture_dataset: GestureDataset = GestureDataset(Path('sl_ai/ai_data/vgt-all'))
gesture_dataset.load_from_csv(Path('sl_ai/gestures_dataset.csv'))

# TODO: Handedness should be retrieved from the django database.
# handedness_data = {}
# for gesture_folder in os.listdir(gesture_dataset.dataset_location):
#     gesture_name, handedness_string = gesture_folder.split('_')
#     handedness_data[gesture_name] = (handedness_string[0] == '1', handedness_string[1] == '1')

classifier: GestureClassifier = GestureClassifier(gesture_dataset=gesture_dataset)
classifier.load_saved_model(model_path=Path('sl_ai/model.h5'))
classifier.summary()
