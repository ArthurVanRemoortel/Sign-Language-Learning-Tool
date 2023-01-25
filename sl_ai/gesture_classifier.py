import os
from typing import Optional

import numpy as np
from pathlib import Path

import pandas as pd
import seaborn as sns
from keras.saving.legacy.save import load_model

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sl_ai.dataset import GestureDataset, fill_holes, make_coordinates_list_fixed_length, pre_process_point_history_center
from sl_ai.config import MAX_VIDEO_FRAMES
import tensorflow as tf
from keras.models import load_model, save_model


class GestureClassifier:
    def __init__(
            self, gesture_dataset: GestureDataset
    ):
        self.gesture_dataset: GestureDataset = gesture_dataset
        self.model = None
        self.train_history = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self, new_dataset: GestureDataset):
        self.gesture_dataset = new_dataset

    def append_dataset(self, new_dataset: GestureDataset):
        self.gesture_dataset.append_dataset(new_dataset)

    def make_model(self):
        TIME_STEPS = MAX_VIDEO_FRAMES * 2 * 2
        DIMENSION = 1
        NUM_CLASSES = len(self.gesture_dataset)
        # TODO: Experiment with tensorflow optimisers.
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(TIME_STEPS, )),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        # self.model = tf.keras.models.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=(TIME_STEPS, DIMENSION)),
        #     tf.keras.layers.Reshape((TIME_STEPS, DIMENSION), input_shape=(TIME_STEPS * DIMENSION,)),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.LSTM(16, input_shape=[TIME_STEPS, DIMENSION]),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.Dense(10, activation='relu'),
        #     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        # ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Experiment using different loss and metric functions.
            metrics=['sparse_categorical_accuracy']
        )

    def train(self, save_path: Optional[Path] = None, train_size=0.5):
        if len(self.gesture_dataset) == 0:
            raise Exception('Tried to train but the dataset is empty.')
        print(f"Training model:")
        self.gesture_dataset.summary()

        # if not self.model:
        self.make_model()

        # TODO: Make sure all categories are represented.
        self.x_train, x_test, self.y_train, y_test = train_test_split(self.gesture_dataset.x_data, self.gesture_dataset.y_data, train_size=train_size, random_state=42, stratify=self.gesture_dataset.y_data, shuffle=True)
        x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, shuffle=True, stratify=y_test)
        self.x_test = x_test
        self.y_test = y_test

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, verbose=1, save_weights_only=True, save_best_only=True)
        es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1)
        self.train_history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1000,
            batch_size=1,
            validation_data=(x_validate, y_validate),
            callbacks=[es_callback]
        )
        [loss, acc] = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("Accuracy:" + str(acc))
        print("Loss:" + str(loss))

        if save_path:
            self.save_model(save_path)

    def save_model(self, save_path):
        if not self.model:
            raise Exception('Cannot save model that has not been created yet')
        self.model.save(save_path)

    def load_saved_model(self, model_path: Path):
        if not model_path.exists():
            raise Exception(f'No model found at {model_path}')
        self.model = load_model(model_path)

    def summary(self):
        if not self.model:
            raise Exception('Cannot make summary of model when it not created yet.')
        return self.model.summary()

    def predict(self, left_landmarks, right_landmarks):
        if not self.model:
            raise Exception('Cannot make prediction when the model is not loaded yet.')

        left_landmarks = np.array(left_landmarks[12], dtype='float32')
        right_landmarks = np.array(right_landmarks[12], dtype='float32')

        x_data = np.concatenate((left_landmarks, right_landmarks), axis=0)#.astype(dtype='float32')
        x_data = x_data.reshape((-1, x_data.shape[0]))
        prediction = self.model.predict(x_data)
        return prediction

    def visualize_accuracy(self):
        """
        Visualize model accuracy
        """
        if not self.train_history:
            raise Exception('Cannot visualise model without training history.')

        plt.plot(self.train_history.history['sparse_categorical_accuracy'], label='training accuracy')
        plt.plot(self.train_history.history['val_sparse_categorical_accuracy'], label='testing accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()

    def visualize_loss(self):
        """
        Visualize model loss
        """
        if not self.train_history:
            raise Exception('Cannot visualise model without training history.')

        plt.plot(self.train_history.history['loss'], label='training loss')
        plt.plot(self.train_history.history['val_loss'], label='validation loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()

    def confusion_matrix(self):
        y_pred = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(self.y_test, y_pred))

        labels = sorted(list(set(self.y_test)))
        cmx_data = confusion_matrix(self.y_test, y_pred, labels=labels)

        df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
        ax.set_ylim(len(set(self.y_test)), 0)
        plt.show()



if __name__ == '__main__':
    CSV_OUT_PATH = Path('gestures_dataset.csv')
    DATASET_LOCATION = Path('ai_data/vgt-all')

    gesture_dataset: GestureDataset = GestureDataset()
    # gesture_dataset.analyze_videos(CSV_OUT_PATH, overwrite=True)
    gesture_dataset.load_from_csv(CSV_OUT_PATH)

    classifier: GestureClassifier = GestureClassifier(gesture_dataset=gesture_dataset)
    classifier.train(save_path=Path("model.h5"))
    classifier.summary()
