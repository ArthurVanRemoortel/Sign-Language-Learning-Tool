import os
from typing import Optional

import numpy as np
from pathlib import Path

import pandas as pd
import seaborn as sns
from keras.regularizers import L2
from keras.saving.legacy.save import load_model

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sl_ai.dataset import (
    GestureDataset,
    fill_holes,
    make_coordinates_list_fixed_length,
    pre_process_point_history_center,
)
from sl_ai.config import MAX_VIDEO_FRAMES, ONLY_LANDMARK_ID
import tensorflow as tf
from keras.models import load_model, save_model
import keras.optimizers

from sl_ai.utils import clean_listdir


class GestureClassifier:
    def __init__(self, gesture_dataset: GestureDataset):
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

    def update_gesture_dataset(self, new_dataset: GestureDataset):
        self.gesture_dataset.update_gesture_dataset(new_dataset)

    def make_model(self, type="lstm"):
        TIME_STEPS = MAX_VIDEO_FRAMES * 2 * 2
        DIMENSION = 1
        NUM_CLASSES = len(self.gesture_dataset)
        # TODO: Experiment with tensorflow optimisers.
        print("Training on shape:", self.x_train[0].shape)
        print("Dateset shape:", self.x_train.shape)
        if type == "lstm":
            self.model = tf.keras.models.Sequential(
                [
                    # tf.keras.layers.InputLayer(
                    #     input_shape=(self.x_train.shape[1], self.x_train.shape[2])
                    # ),
                    tf.keras.layers.LSTM(64, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), return_sequences=True, kernel_regularizer=L2(0.001)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=L2(0.001)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
                ]
            )
        elif type == "standard":
            self.model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=self.x_train[0].shape),
                    tf.keras.layers.Flatten(),
                    # tf.keras.layers.Dense(64, activation="relu", input_shape=self.x_train[0].shape),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
                ]
            )
        else:
            raise Exception(f"Unknown model type {type}")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",  # Experiment using different loss and metric functions.
            metrics=["sparse_categorical_accuracy"],
        )

    def train(self, save_path: Optional[Path] = None, train_size=0.5, type="lstm"):
        if len(self.gesture_dataset) == 0:
            raise Exception("Tried to train but the dataset is empty.")
        self.gesture_dataset.summary()

        # TODO: Make sure all categories are represented.
        x = []
        if type == "standard":
            print(len(self.gesture_dataset.x_data[0]))
            for row in self.gesture_dataset.x_data:
                new_row = [row[0]+row[1], row[2]+row[3]]
                # for n in range(0, 4, 2):
                #     print(f"Selecting feature {n}")
                #     new_row.append(row[n] + row[n+1])
                x.append(new_row)
            x = np.array(x, dtype=np.float)
            x = np.array(self.gesture_dataset.x_data, dtype=np.float)
        elif type == "lstm":
            x = np.array(self.gesture_dataset.x_data, dtype=np.float)
        else:
            raise Exception(f"Unknown model type provided: {type}")

        self.x_train, x_test, self.y_train, y_test = train_test_split(
            x,
            self.gesture_dataset.y_data,
            train_size=train_size,
            random_state=42,
            stratify=self.gesture_dataset.y_data,
            shuffle=True,
        )
        x_validate, x_test, y_validate, y_test = train_test_split(
            x_test,
            y_test,
            test_size=0.5,
            random_state=42,
            shuffle=True,
            stratify=y_test,
        )
        self.x_test = x_test
        self.y_test = y_test

        self.make_model(type=type)

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, verbose=1, save_weights_only=True, save_best_only=True)
        es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1)
        self.train_history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1000,
            batch_size=1,
            validation_data=(x_validate, y_validate),
            callbacks=[es_callback],
        )
        [loss, acc] = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("Accuracy:" + str(acc))
        print("Loss:" + str(loss))

        if save_path:
            self.save_model(save_path)

    def save_model(self, save_path):
        if not self.model:
            raise Exception("Cannot save model that has not been created yet")
        self.model.save(save_path)

    def load_saved_model(self, model_path: Path):
        if not model_path.exists():
            raise Exception(f"No model found at {model_path}")
        self.model = load_model(model_path)

    def summary(self):
        if not self.model:
            raise Exception("Cannot make summary of model when it not created yet.")
        return self.model.summary()

    def predict(self, left_x, left_y, right_x, right_y, left_angles, right_angles, left_openness, right_openness):
        if not self.model:
            raise Exception("Cannot make prediction when the model is not loaded yet.")
        # print(right_landmarks.keys())
        # left_landmarks = np.array(left_landmarks[ONLY_LANDMARK_ID], dtype="float32")
        # right_landmarks = np.array(right_landmarks[ONLY_LANDMARK_ID], dtype="float32")
        # # left_landmarks0 = np.array(left_landmarks[0], dtype="float32")
        # # right_landmarks0 = np.array(right_landmarks[0], dtype="float32")
        # x_data = np.concatenate(
        #     (left_landmarks, right_landmarks), axis=0
        # )
        # x_data = x_data.reshape((-1, x_data.shape[0]))
        #
        # other_data = np.concatenate(
        #     (left_landmarks, right_landmarks), axis=0
        # )
        # other_data = other_data.reshape((-1, other_data.shape[0]))
        # d = [x_data, *other_data]
        # print("SHAPE", d.shape)
        # x_data = x_data.reshape((-1, x_data.shape[0]))
        # other_data = other_data.reshape((-1, other_data.shape[0]))
        # d = np.array(*d)
        # print("SHAPE: ", len(d))
        x_data = [
            left_x,
            left_y,
            right_x,
            right_y,
            left_angles,
            right_angles,
            *left_openness,
            *right_openness
        ]
        for item in x_data:
            print(len(item), type(item))
        # x_data = [*landmark_history]
        # print(len(landmark_history))
        prediction = self.model.predict([x_data])
        return prediction

    def visualize_accuracy(self):
        """
        Visualize model accuracy
        """
        if not self.train_history:
            raise Exception("Cannot visualise model without training history.")

        plt.plot(
            self.train_history.history["sparse_categorical_accuracy"],
            label="training accuracy",
            color="blue",
        )
        plt.plot(
            self.train_history.history["val_sparse_categorical_accuracy"],
            label="testing accuracy",
            color="green",
        )
        plt.title("Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()

    def visualize_loss(self):
        """
        Visualize model loss
        """
        if not self.train_history:
            raise Exception("Cannot visualise model without training history.")

        plt.plot(
            self.train_history.history["loss"], label="training loss", color="yellow"
        )
        plt.plot(
            self.train_history.history["val_loss"],
            label="validation loss",
            color="orange",
        )
        plt.title("Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()

    def confusion_matrix(self):
        y_pred = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred, axis=1)
        # print(classification_report(self.y_test, y_pred))
        labels = sorted(list(set(self.y_test)))
        labels_text = [self.gesture_dataset.lookup_dict[id] for id in labels]
        cmx_data = confusion_matrix(self.y_test, y_pred, labels=labels)
        df_cmx = pd.DataFrame(cmx_data, index=labels_text, columns=labels_text)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df_cmx, annot=True, fmt="g", square=False)
        ax.set_ylim(len(set(self.y_test)), 0)
        plt.show()


if __name__ == "__main__":
    DATASET_LOCATION = Path("ai_data/vgt-all")
    handedness_data = {}
    for gesture_folder in clean_listdir(DATASET_LOCATION):
        *gestures_name_parts, handedness_string = gesture_folder.split("_")
        gesture_name = "_".join(gestures_name_parts)
        handedness_data[gesture_name] = (
            handedness_string[0] == "1",
            handedness_string[1] == "1",
        )

    gesture_dataset: GestureDataset = GestureDataset()
    gesture_dataset.scan_videos(DATASET_LOCATION, handedness_data=handedness_data)
    gesture_dataset.load_gestures_from_csv()

    classifier: GestureClassifier = GestureClassifier(gesture_dataset=gesture_dataset)
    classifier.train(train_size=0.5)
