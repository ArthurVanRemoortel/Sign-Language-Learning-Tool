import copy
import csv
import itertools
import os
from pathlib import Path
from pprint import pprint
from typing import List, Union, Optional

import cv2
import mediapipe as mp
import numpy as np
import skvideo.io
from mediapipe.framework.formats.classification_pb2 import ClassificationList
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sl_ai.config import MAX_VIDEO_FRAMES

# DATASET_LOCATION = Path('sl_ai/ai_data/vgt-all')

USE_STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5


def log_csv(file_path: Path, gesture_name: str, gesture_number, video_number, hand_number, landmark_id, point_history_list):
    with open(file_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_name, gesture_number, video_number, hand_number, landmark_id, *point_history_list])


def fill_holes(data_list: list, empty_val):
    """
    Imputes holes in a list of numbers (coordinates) with a number in the middle of its neighbors.
    """
    previous_value = empty_val
    result = []
    for i, value in enumerate(data_list):
        if value == empty_val:
            if previous_value == empty_val:
                result.append(previous_value)
            else:
                try:
                    rest = data_list[i:]
                    while rest[0] == empty_val:
                        rest = rest[1:]
                    next_x, next_y = rest[0]
                    prev_x, prev_y = previous_value
                    result.append([int((prev_x + next_x) / 2), int((prev_y + next_y) / 2)])
                except IndexError:
                    result.append(previous_value)
        else:
            previous_value = value
            result.append(value)
    return result


def shrink_coordinates_list(lst: list, size: int):
    """
    Shrink a list of numbers (coordinates) to a fixed length.
    When shrinking a list some values will need to be discarded because. Instead of completely discarding these values,
    insert a number inbetween the 2 numbers when they would end up at the same index.
    """
    if len(lst) <= size:
        return copy.deepcopy(lst)

    skip_step = size / len(lst)
    result = [None for _ in range(size)]

    for i, value in enumerate(lst):
        insert_index = int(i * skip_step)
        existing_at_index = result[insert_index]
        if existing_at_index:
            existing_x, existing_y = existing_at_index
            x, y = value
            result[insert_index] = [(existing_x + x) / 2, (existing_y + y) / 2]
        else:
            result[insert_index] = value
    return result


def extend_coordinates_list(lst: list, size: int) -> list:
    if len(lst) >= size:
        return copy.deepcopy(lst)
    skip_step = size / len(lst)
    result = [None for _ in range(size)]
    for i, value in enumerate(lst):
        insert_index = int(i * skip_step)
        result[insert_index] = value

    check_end_i = -1
    while result[check_end_i] is None:
        result[check_end_i] = lst[-1]
        check_end_i -= 1

    prev_x = 0
    prev_y = 0
    for i, item in enumerate(result):
        if not item:
            # fil it.
            # find next value.
            rest = [v for v in result[i+1:]]
            while not rest[0]:
                rest = rest[1:]
            next_x, next_y = rest[0]
            between_x = (prev_x + next_x) / 2
            between_y = (prev_y + next_y) / 2
            result[i] = [between_x, between_y]
        else:
            prev_x, prev_y = item
    return result


def make_coordinates_list_fixed_length(lst: (float, float), size: int) -> list:
    lst_len = len(lst)
    if lst_len == size:
        return copy.deepcopy(lst)
    elif lst_len > size:
        return shrink_coordinates_list(lst, size)
    else:
        return extend_coordinates_list(lst, size)


def read_video(file_path: Union[str, Path], as_grey=False):
    video_data = skvideo.io.vreader(str(file_path), as_grey=as_grey)
    return list(video_data)


def is_landmark_in_active_zone(landmarks):
    ys = [l.y for l in list(landmarks)]
    return min(ys) <= 0.85


def calculate_landmark_list(image_width, image_height, landmarks):
    """
    Original from https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    Converts the x and y coordinates between 0 -> 1 to image_width -> image_height
    Expects the 21 landmarks of the hand as input.
    """
    if isinstance(landmarks, list):
        x_getter = lambda landmark: landmark['x']
        y_getter = lambda landmark: landmark['y']
    else:
        x_getter = lambda landmark: landmark.x
        y_getter = lambda landmark: landmark.y

    landmark_point = []
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(x_getter(landmark) * image_width), image_width - 1)
        landmark_y = min(int(y_getter(landmark) * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_point_history(image_width, image_height, point_history):
    """
    Original from https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    Converts a list of coordinates to a list of deltas corresponding to how much a landmark moves relative to the start on the gesture.
    """
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = image_width / 2, image_height / 2
    for index, point in enumerate(temp_point_history):
        # if index == 0:
        #     base_x, base_y = point[0], point[1]
        # if temp_point_history[index] == [0, 0]:
        #     continue
        if point == [-1, -1]:
            continue
        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))  # [[x, y], [x, y]] => [x, y, x, y]
    return temp_point_history


def trim_landmark_lists(left_landmarks, right_landmarks, empty_val):
    # Remove empty frames at the beginning or end of the video. Usually when the person in the video is standing still.
    for landmark_id, left in left_landmarks.items():
        right = right_landmarks[landmark_id]
        while left[0] == empty_val and right[0] == empty_val:
            left = left[1:]
            right = right[1:]
            # skipped an empty frame in the beginning.
        while left[-1] == empty_val and right[-1] == empty_val:
            left = left[:-1]
            right = right[:-1]
        left_landmarks[landmark_id] = left
        right_landmarks[landmark_id] = right


class GestureData:
    def __init__(self, name, left_hand=True, right_hand=True):
        self.name: str = name
        self.left_hand: bool = left_hand
        self.right_hand: bool = right_hand
        self.reference_videos: List[Path] = []

    def add_video(self, video_path: Path):
        self.reference_videos.append(video_path)

    def uses_hand(self, hand_name: str) -> bool:
        if hand_name.lower() == 'left':
            return self.left_hand
        elif hand_name.lower() == 'right':
            return self.right_hand
        return False

    def __str__(self):
        return f'Gesture: {self.name}'


def preprocess_landmarks(left_landmarks, right_landmarks, frame_width, frame_height):
    trim_landmark_lists(left_landmarks, right_landmarks, [-1, -1])
    for landmark_id, landmarks in left_landmarks.items():
        landmarks = fill_holes(landmarks, [-1, -1])
        landmarks = make_coordinates_list_fixed_length(landmarks, MAX_VIDEO_FRAMES)
        landmarks = pre_process_point_history(frame_width, frame_height, landmarks)
        left_landmarks[landmark_id] = landmarks

    for landmark_id, landmarks in right_landmarks.items():
        landmarks = fill_holes(landmarks, [-1, -1])
        landmarks = make_coordinates_list_fixed_length(landmarks, MAX_VIDEO_FRAMES)
        # print(landmarks)
        landmarks = pre_process_point_history(frame_width, frame_height, landmarks)
        # print(landmarks)
        # print()
        right_landmarks[landmark_id] = landmarks


class GestureDataset:
    def __init__(
            self,
    ):
        self.gestures: List[GestureData] = []  # No data if loaded from csv file.

        self.x_data = None
        self.y_data = None

    def summary(self):
        print(f"Dataset contain {len(np.unique(self.y_data))} gestures.")

    def load_from_csv(self, csv_path: Path):
        if not csv_path.exists():
            raise Exception(f"Did not find the csv dataset at {csv_path}")
        X_dataset = []
        Y_dataset = []
        gesture_id = -1
        last_gesture_number = -1
        with open(csv_path, 'r', encoding="utf-8") as data_file:
            lines = data_file.readlines()
            for landmark_line in lines:
                landmark_line = landmark_line.split(',')
                # TODO: Add handedness to each row.
                gesture_name, gesture_number, video_number, hand_number, landmark_id, *history = landmark_line
                gesture_number = int(gesture_number)
                video_number = int(video_number)
                hand_number = int(hand_number)
                landmark_id = int(landmark_id)

                if csv_path.name == "gestures_dataset.csv" and gesture_name not in ["belgië", "verenigde staten", "hallo"]:  # Currently only these gestures have enough data. TODO: Remove this later.
                    continue

                if landmark_id != 0:
                    continue

                if gesture_number != last_gesture_number:
                    last_gesture_number = gesture_number
                    gesture_id += 1

                print(f"{gesture_name}({gesture_number} -> {gesture_id})")

                history = np.array(history, dtype='float32')
                if hand_number == 0:
                    Y_dataset.append(gesture_id)
                    X_dataset.append(history)
                elif hand_number == 1:
                    X_dataset[-1] = np.append(X_dataset[-1], history)  # Appends the right hand after the left hand.

        self.y_data = np.array(Y_dataset)
        self.x_data = np.array(X_dataset)

    def append_dataset(self, other_dataset: 'GestureDataset'):
        if len(other_dataset.y_data) == 0:
            return
        gesture_id_base = np.amax(self.y_data) + 1
        new_y_data = other_dataset.y_data + gesture_id_base
        new_x_data = other_dataset.x_data
        self.y_data = np.concatenate([self.y_data, new_y_data])
        self.x_data = np.concatenate([self.x_data, new_x_data])




    def analyze_videos(self, csv_out_path: Optional[Path] = None, overwrite=False):
        """
        Use mediapipe to detect hand landmarks in every training video and save this data in a usable format.
        TODO: Could use some serious refactoring.
        TODO: Use multiprocessing.
        """
        print(f"Analysing video {sum([len(g.reference_videos) for g in self.gestures])} files.")
        if csv_out_path and overwrite and csv_out_path.exists():
            os.remove(csv_out_path)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=USE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=1
        )
        for gesture_i, gesture in enumerate(tqdm(self.gestures)):
            # Loop over all gestures in the dataset.
            for video_i, video_path in enumerate(gesture.reference_videos):
                # Loop over all the videos per gesture.
                video_data = read_video(video_path)
                frames_number = len(video_data)

                # Dictionaries for every hand.
                left_landmarks = {i: [] for i in range(0, 21)}
                right_landmarks = {i: [] for i in range(0, 21)}


                frame_height = None
                frame_width = None

                for frame_i, frame in enumerate(video_data):
                    # Detect hands in every frame of the video.
                    if frame_height is None:
                        frame_height, frame_width, _ = frame.shape
                    frame = cv2.flip(frame, 1)  # Mediapipe was designed to work with webcam video, which is mirrored. The videos in the dataset are not.
                    frame.flags.writeable = False
                    results = hands.process(frame)

                    if not results.multi_hand_landmarks:
                        # Nothing was detected this frame.
                        for landmark_id in left_landmarks.keys():
                            left_landmarks[landmark_id].append([-1, -1])
                        for landmark_id in right_landmarks.keys():
                            right_landmarks[landmark_id].append([-1, -1])
                    else:
                        # Found some hands.
                        found_left = False
                        found_right = False
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                              results.multi_handedness):
                            hand_name = handedness.classification[0].label.lower()
                            landmarks_coordinates = hand_landmarks.landmark
                            if not gesture.uses_hand(hand_name):
                                # Detected a hand that should not have been tracked.
                                continue
                            if not is_landmark_in_active_zone(hand_landmarks.landmark):
                                # Ignore when the hands are at the edge of the frame. The person is the video is still
                                # getting in position and is not performing the gesture yet.
                                continue
                            landmark_list = calculate_landmark_list(frame_width, frame_height, landmarks_coordinates)
                            for landmark_id, landmark in enumerate(landmark_list):
                                if hand_name == 'left':
                                    left_landmarks[landmark_id].append(landmark)
                                    # right_landmarks[landmark_id].append([-1, -1])
                                    found_left = True
                                elif hand_name == 'right':
                                    right_landmarks[landmark_id].append(landmark)
                                    # left_landmarks[landmark_id].append([-1, -1])
                                    found_right = True
                        if not found_left:
                            for landmark_id in left_landmarks.keys():
                                left_landmarks[landmark_id].append([-1, -1])
                        if not found_right:
                            for landmark_id in right_landmarks.keys():
                                right_landmarks[landmark_id].append([-1, -1])

                preprocess_landmarks(left_landmarks, right_landmarks, frame_width, frame_height)

                if csv_out_path:
                    for landmark_id, values in left_landmarks.items():
                        log_csv(csv_out_path, gesture_name=gesture.name, gesture_number=gesture_i, video_number=video_i, hand_number=0, landmark_id=landmark_id,
                                point_history_list=values)

                    for landmark_id, values in right_landmarks.items():
                        log_csv(csv_out_path, gesture_name=gesture.name, gesture_number=gesture_i, video_number=video_i, hand_number=1, landmark_id=landmark_id,
                                point_history_list=values)

    def scan_videos(self, dataset_location: Path, handedness_data):
        self.gestures.clear()
        for gesture_folder in os.listdir(dataset_location):
            gesture_name = gesture_folder.split('_')[0]
            gesture_path = dataset_location / str(gesture_folder)
            left_hand, right_hand = handedness_data[gesture_name]
            gesture = GestureData(name=gesture_name, left_hand=left_hand, right_hand=right_hand)
            for video_name in os.listdir(gesture_path):
                gesture.add_video(gesture_path / str(video_name))
            self.gestures.append(gesture)
        print(f'Loaded {len(self.gestures)} gestures')

    def add_django_gesture(self, django_gesture: 'Gesture'):
        gesture_data = GestureData(name=django_gesture.word, left_hand=django_gesture.left_hand, right_hand=django_gesture.right_hand)
        gesture_path = Path('sl_ai/ai_data/vgt-uploaded')
        if django_gesture.creator:
            gesture_path = gesture_path / str(django_gesture.creator.id)
        gesture_path = gesture_path / django_gesture.handed_string
        for video_name in os.listdir(gesture_path):
            gesture_data.add_video(gesture_path / str(video_name))
        self.gestures.append(gesture_data)


    def __len__(self):
        return len(np.unique(self.y_data))


if __name__ == '__main__':
    CSV_OUT_PATH = Path('gestures_dataset.csv')
    DATASET_LOCATION = Path('ai_data/vgt-all')

    UPLOADED_CSV_OUT_PATH = Path('gestures_dataset.csv')
    UPLOADED_DATASET_LOCATION = Path('ai_data/vgt-all')

    # dataset = GestureDataset()
    # handedness_data = {}
    # for gesture_folder in os.listdir(DATASET_LOCATION):
    #     gesture_name, handedness_string = gesture_folder.split('_')
    #     handedness_data[gesture_name] = (handedness_string[0] == '1', handedness_string[1] == '1')
    #
    # dataset.scan_videos(dataset_location=DATASET_LOCATION, handedness_data=handedness_data)
    # # dataset.analyze_videos(csv_out_path=CSV_OUT_PATH, overwrite=True)
    # dataset.load_from_csv(CSV_OUT_PATH)


    dataset = GestureDataset()
    handedness_data = {}
    for gesture_folder in os.listdir(UPLOADED_DATASET_LOCATION):
        gesture_name, handedness_string = gesture_folder.split('_')
        handedness_data[gesture_name] = (handedness_string[0] == '1', handedness_string[1] == '1')

    dataset.scan_videos(dataset_location=UPLOADED_DATASET_LOCATION, handedness_data=handedness_data)
    # dataset.analyze_videos(csv_out_path=CSV_OUT_PATH, overwrite=True)
    dataset.load_from_csv(UPLOADED_CSV_OUT_PATH)