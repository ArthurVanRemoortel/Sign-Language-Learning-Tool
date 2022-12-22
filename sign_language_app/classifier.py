import os
from pathlib import Path

from sl_ai.dataset import GestureDataset
from sl_ai.gesture_classifier import GestureClassifier

gesture_dataset: GestureDataset = GestureDataset()
gesture_dataset.load_from_csv(Path('sl_ai/gestures_dataset.csv'))

uploaded_dataset: GestureDataset = GestureDataset()
uploaded_dataset.load_from_csv(Path('sl_ai/uploaded_gestures_dataset.csv'))

gesture_dataset.append_dataset(uploaded_dataset)
gesture_dataset.summary()

gesture_classifier: GestureClassifier = GestureClassifier(gesture_dataset=gesture_dataset)

model_path = Path('sl_ai/model.h5')
if model_path.exists():
    gesture_classifier.load_saved_model(model_path=model_path)
else:
    print('No model file found. Training a new model.')
    gesture_classifier.train(save_path=model_path)

gesture_classifier.summary()

# gesture_classifier = None