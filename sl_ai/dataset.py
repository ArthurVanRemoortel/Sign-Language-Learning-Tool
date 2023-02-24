import copy
import csv
import functools
import itertools
import math
import os
import time
from ast import literal_eval
from math import sqrt
from pathlib import Path
from pprint import pprint
from typing import List, Union, Optional, Tuple, Dict, Any, Callable

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import skvideo.io
from matplotlib import pyplot as plt
from tqdm import tqdm

from sl_ai.config import MAX_VIDEO_FRAMES, ONLY_LANDMARK_ID
from multiprocessing import Pool

from sl_ai.utils import clean_listdir, is_video

# DATASET_LOCATION = Path('sl_ai/ai_data/vgt-all')


USE_STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.8


def log_csv(
    writer,
    gesture_name: str,
    gesture_number,
    video_number,
    hand_number,
    landmark_id,
    mouth_position,
    point_history_list,
    width,
    height,
):
    """Saves data to a csv file"""
    writer.writerow(
        [
            gesture_name,
            gesture_number,
            video_number,
            width,
            height,
            hand_number,
            landmark_id,
            mouth_position,
            point_list_to_string(point_history_list),
        ]
    )


def point_list_to_string(lst) -> str:
    string = ""
    for i, item in enumerate(lst):
        if i != 0:
            string += ", "
        string += str(item).replace(" ", "")
    string = f"[{string}]"
    return string


def point_list_from_string(string: str) -> list:
    result = []
    for item in string[1:-1].split(", "):
        item = item[1:-1]
        numbers = []
        for number in item.split(","):
            numbers.append(float(number))
        result.append(numbers)
    return result


def fill_holes(data_list: List[Tuple[float]], empty_val: Any) -> List[Tuple[float]]:
    """
    Fill holes in a list of numbers (coordinates) with a number in the middle of its neighbors.
    empty_val: Determines what is considered a missing value.
    Example [(1, 1), (None, None) (3, 3)] -> [(1, 1), (2, 2) (3, 3)]
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
                    new_value = [((prev_x + next_x) / 2), ((prev_y + next_y) / 2)]
                    result.append(new_value)
                    previous_value = new_value
                except IndexError:
                    result.append(previous_value)
                    previous_value = value
        else:
            previous_value = value
            result.append(value)
    return result


def shrink_list(lst: [], new_size: int, combine_function: Callable) -> []:
    """
    Shrink a list of numbers (or any oject) to a fixed length.
    When shrinking a list some values will need to be discarded because they will want to end up at the same location.
    Instead of completely discarding these values, insert a number inbetween the 2 numbers when they would end up at the same index.
    combine_function determines how a value is calculated when they overlap. (e.g take the average of the 2 numbers.)
    Example: [1, 2, 4, 5] -> [1, 3, 5]
    """
    if len(lst) <= new_size:
        return copy.deepcopy(lst)
    skip_step = new_size / len(lst)
    result = [None for _ in range(new_size)]
    for i, value in enumerate(lst):
        insert_index = int(i * skip_step)
        existing_at_index = result[insert_index]
        if existing_at_index:
            existing_value = existing_at_index
            new_value = combine_function(existing_value, value)
            result[insert_index] = new_value
        else:
            result[insert_index] = value
    return result


def extend_list(lst: [], new_size: int, create_function: Callable) -> []:
    """
    Extends a list. This will create empty values that will need to filled up using the average of its neighbors.
    Example: [1, 5] -> [1, 3, 5]
    """
    if len(lst) >= new_size:
        return copy.deepcopy(lst)
    skip_step = new_size / len(lst)
    result = [None for _ in range(new_size)]
    for i, value in enumerate(lst):
        insert_index = int(i * skip_step)
        result[insert_index] = value
    check_end_i = -1
    while result[check_end_i] is None:
        result[check_end_i] = lst[-1]
        check_end_i -= 1

    prev_value = 0
    for i, item in enumerate(result):
        if not item:
            rest = [v for v in result[i + 1 :]]
            while len(rest) > 0 and rest[0] is None:
                rest = rest[1:]
            if not rest:
                next_value = prev_value
            else:
                next_value = rest[0]
            new_value = create_function(prev_value, next_value)
            result[i] = new_value
        else:
            prev_value = item
    return result


def extend_coordinates_list(
    lst: List[Tuple[float]], new_size: int
) -> List[Tuple[float]]:
    """
    Extends a list. This will create empty values that will need to filled up using the average of its neighbors.
    Example: [1, 5] -> [1, 3, 5]
    """
    if len(lst) >= new_size:
        return copy.deepcopy(lst)
    skip_step = new_size / len(lst)
    result = [None for _ in range(new_size)]
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
            rest = [v for v in result[i + 1 :]]
            while not rest[0]:
                rest = rest[1:]
            next_x, next_y = rest[0]
            between_x = (prev_x + next_x) / 2
            between_y = (prev_y + next_y) / 2
            result[i] = [between_x, between_y]
        else:
            prev_x, prev_y = item
    return result


def make_coordinates_list_fixed_length(lst: List[Tuple[float]], new_size: int) -> List:
    """
    Uses extend_list and shrink_list to resize a list to a specific size.
    """
    return make_list_fixed_length(
        lst,
        new_size,
        combine_function=lambda existing, value: [
            (existing[0] + value[0]) / 2,
            (existing[1] + value[1]) / 2,
        ],
        create_function=lambda prev_value, next_value: [
            (prev_value[0] + next_value[0]) / 2,
            (prev_value[1] + next_value[1]) / 2,
        ],
    )


def make_list_fixed_length(
    lst: List[Tuple[float]],
    new_size: int,
    combine_function: Callable,
    create_function: Callable,
) -> List:
    lst_len = len(lst)
    if lst_len == new_size:
        return copy.deepcopy(lst)
    elif lst_len > new_size:
        return shrink_list(lst, new_size, combine_function=combine_function)
    else:
        return extend_list(
            lst,
            new_size,
            create_function=create_function,
        )


def read_video(file_path: Union[str, Path], as_grey=False) -> List:
    """Read a video file and returns a 3D array frame and pixels."""
    video_data = skvideo.io.vreader(str(file_path), as_grey=as_grey)
    return list(video_data)


def is_landmark_in_active_zone(landmarks):
    """
    Can be used to ignore gestures whet they are close to the edge of the screen.
    Usually these movements are not part of the gesture itself but just the hand entering and leaving the screen.
    (currently not used)
    """
    ys = [l.y for l in list(landmarks)]
    return min(ys) <= 0.85


def mirror_landmarks_list(landmarks: List[List[float]]) -> List[List[float]]:
    """
    Mirrors the x coordinates of all landmarks.
    Data coming in from the webcam/browser is mirrored and will not match the dataset videos.
    """
    mirrored_landmarks = []
    for _, landmark in enumerate(landmarks):
        mirrored_landmarks.append(mirror_coordinate(landmark[0], landmark[1]))
    return mirrored_landmarks


def mirror_coordinate(x, y):
    return [1 - x, y]


def calculate_landmark_list(
    image_width: int, image_height: int, landmarks
) -> List[List[Any]]:
    """
    Original from https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    Selects the relevant data from all the data generated by the mediapipe library.
    Currently, simply convert mediapipe objects to a standatd python dictionary containing lists.
    """
    if isinstance(landmarks, list):
        x_getter = lambda landmark: landmark["x"]
        y_getter = lambda landmark: landmark["y"]
    else:
        x_getter = lambda landmark: landmark.x
        y_getter = lambda landmark: landmark.y

    landmark_point = []
    for _, landmark in enumerate(landmarks):
        landmark_x = x_getter(landmark)
        landmark_y = y_getter(landmark)
        landmark_x = round(landmark_x, 3)
        landmark_y = round(landmark_y, 3)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_point_history_center(
    image_width: int, image_height: int, point_history: List[Tuple[float]]
) -> List[float]:
    """Alternative data representation for the AI model. Converts a list of coordinates to a list of distances to the center of the screen."""
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0.5, 0.5
    for index, point in enumerate(temp_point_history):
        if point == [-1, -1] or point == [-1.0, -1.0]:
            continue
        temp_point_history[index][0] = temp_point_history[index][0] - base_x
        temp_point_history[index][1] = temp_point_history[index][1] - base_y
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history)
    )  # [[x, y], [x, y]] => [x, y, x, y] # TODO: Make this a separate function.
    return temp_point_history


def pre_process_point_history_mouth_position(
    mouth_position: List[float], point_history: List[List[float]]
) -> (List[float], List[float]):
    """Alternative data representation for the AI model. Converts a list of coordinates to a list of distances relative to the position of the users mouth."""
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = mouth_position
    for index, point in enumerate(temp_point_history):
        if point == [-1, -1] or point == [-1.0, -1.0]:
            continue
        temp_point_history[index][0] = temp_point_history[index][0] - base_x
        temp_point_history[index][1] = temp_point_history[index][1] - base_y
    # temp_point_history = list(
    #     itertools.chain.from_iterable(temp_point_history)
    # )  # [[x, y], [x, y]] => [x, y, x, y] # TODO: Make this a separate function.
    x_coordinates = []
    y_coordinates = []
    for coordinate in temp_point_history:
        x_coordinates.append(coordinate[0])
        y_coordinates.append(coordinate[1])
    return x_coordinates, y_coordinates
    # return temp_point_history


def pre_process_point_history_deltas(
    image_width: int, image_height: int, point_history: List[Tuple[float]]
) -> List[float]:
    """Alternative data representation for the AI model. Converts a list of coordinates to a list of deltas between the coordinates."""
    temp_point_history = copy.deepcopy(point_history)
    prev_x, prev_y = None, None
    for index, point in enumerate(temp_point_history):
        if point == [-1.0, -1.0]:
            temp_point_history[index] = (0, 0)
            continue
        if prev_x is None and prev_y is None:
            prev_x, prev_y = point
        if temp_point_history[index] == [0, 0]:
            continue
        delta = ((point[0] - prev_x), (point[1] - prev_y))
        temp_point_history[index] = delta
        prev_x, prev_y = point
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history


def trim_landmark_lists(
    left_landmarks: Dict[int, List[float]],
    right_landmarks: Dict[int, List[float]],
    empty_val: Any,
):
    """Remove empty frames at the beginning or end of the video. Usually when the person in the video is standing still"""
    for landmark_id, left in left_landmarks.items():
        right = right_landmarks[landmark_id]
        while left[0] == empty_val and right[0] == empty_val:
            left = left[1:]
            right = right[1:]
        while left[-1] == empty_val and right[-1] == empty_val:
            left = left[:-1]
            right = right[:-1]
        left_landmarks[landmark_id] = left
        right_landmarks[landmark_id] = right


def visualize_gesture(
    coordinates: List[Tuple[float]], frame_width: int, frame_height: int
):
    """Helper function to visualize a list of coordinates in a graph."""
    fixed_coordinates = list(
        map(
            lambda pair: (pair[0] * frame_width, pair[1] * frame_height)
            if pair[0] != -1
            else (0, 0),
            coordinates,
        )
    )
    plt.scatter(*zip(*fixed_coordinates))
    plt.xlim([0, frame_width])
    plt.ylim([0, frame_height])


def preprocess_landmarks(
    left_landmarks: Dict[int, List[Tuple[float]]],
    right_landmarks: Dict[int, List[Tuple[float]]],
    frame_width: int,
    frame_height: int,
):
    """Uses the functions above to convert the raw data from mediapipe library in something usable by the AI model."""
    trim_landmark_lists(left_landmarks, right_landmarks, [-1.0, -1.0])
    for landmark_id, landmarks in left_landmarks.items():
        landmarks = make_coordinates_list_fixed_length(landmarks, MAX_VIDEO_FRAMES)
        landmarks = fill_holes(landmarks, [-1.0, -1.0])
        # landmarks = pre_process_point_history(frame_width, frame_height, landmarks)
        left_landmarks[landmark_id] = landmarks

    for landmark_id, landmarks in right_landmarks.items():
        landmarks = make_coordinates_list_fixed_length(landmarks, MAX_VIDEO_FRAMES)
        landmarks = fill_holes(landmarks, [-1.0, -1.0])
        # landmarks = pre_process_point_history(frame_width, frame_height, landmarks)
        right_landmarks[landmark_id] = landmarks


def process_orientation(landmarks) -> [float]:
    hand_base = landmarks[0]
    hand_center_knuckle = landmarks[9]
    angles = []
    for coordinates in zip(hand_base, hand_center_knuckle):
        base = coordinates[0]
        tip = coordinates[1]
        if (
            tip == [-1, -1]
            or tip == [-1.0, -1.0]
            or base == [-1, -1]
            or base == [-1.0, -1.0]
        ):
            angles.append(0)
            continue
        angle = calculate_angle(*base, *tip)
        angles.append(angle)
    return angles


def calculate_distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def calculate_angle(x0, y0, x1, y1):
    angle = math.atan2(y0 - y1, x0 - x1)
    return np.degrees(angle) % 360.0


def calculate_center(x0, y0, x1, y1) -> [float, float]:
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    return [center_x, center_y]


def scale_number(number, number_min, number_max, new_min=0, new_max=1):
    number_range = number_max - number_min
    new_range = new_max - new_min
    return (((number - number_min) * new_range) / number_range) + new_min


def is_angle_horizontal(angle):
    return 315 < angle < 360 or 0 < angle < 45 or 150 < angle < 210


def hand_openness(landmarks, fingers_tips=None) -> [float]:
    """Not completed"""
    if fingers_tips is None:
        fingers_tips = [8, 12, 16]
    hand_center_landmarks = landmarks[9]
    hand_base_landmarks = landmarks[0]
    finger_landmarks = [
        (landmarks[i], landmarks[j], landmarks[k])
        for i, j, k in [(tip - 3, tip - 2, tip) for tip in fingers_tips]
    ]
    openness = [[] for _ in fingers_tips]
    for frame, center_top_pos in enumerate(hand_center_landmarks):
        if (
                center_top_pos == [-1, -1]
                or center_top_pos == [-1.0, -1.0]
        ):
            for finger_n, _ in enumerate(fingers_tips):
                openness[finger_n].append(0)
            continue

        hand_base_pos = hand_base_landmarks[frame]
        palm_size = calculate_distance(*hand_base_pos, *center_top_pos)
        hand_size = palm_size + calculate_distance(
            *landmarks[9][frame], *landmarks[10][frame]
        )
        frame_fingers_openness = []
        for finger_base, finger_first_knuckle, finger_tip in finger_landmarks:
            finger_tip_pos = finger_tip[frame]
            finger_base_pos = finger_base[frame]
            distance = calculate_distance(*finger_base_pos, *finger_tip_pos)
            relative_distance = distance / hand_size
            scaled_distance = scale_number(relative_distance, 0.1, 0.6)
            scaled_distance = max(0.0, min(scaled_distance, 1.0))
            frame_fingers_openness.append(scaled_distance)
            # print(distance)
        for finger_n, value in enumerate(frame_fingers_openness):
            openness[finger_n].append(value)
    return openness


def make_hands_detector() -> (
    mp.solutions.hands.Hands,
    mp.solutions.face_detection.FaceDetection,
):
    """Creates and instance of the mediapipe hands detector."""
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        model_complexity=1,
    )
    face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    return hands, face


class GestureData:
    # TODO: This class might not be required and is confusing.
    def __init__(
        self, name: str, left_hand: bool, right_hand: bool, dataset_file: Path
    ):
        self.name: str = name
        self.left_hand: bool = left_hand
        self.right_hand: bool = right_hand
        self.reference_videos: List[Path] = []
        self.dataset_file: Path = dataset_file

    def add_video(self, video_path: Path):
        self.reference_videos.append(video_path)

    @property
    def get_folder_name(self) -> str:
        return f"{self.name}_{1 if self.left_hand else 0}{1 if self.right_hand else 0}"

    def uses_hand(self, hand_name: str) -> bool:
        if hand_name.lower() == "left":
            return self.left_hand
        elif hand_name.lower() == "right":
            return self.right_hand
        return False

    def __str__(self):
        return f"Gesture: {self.name} (left={self.left_hand}, right={self.right_hand})"


def detect_hands_task(gesture: GestureData, video_path: Path):
    """Detects gesture landmarks from a video file."""
    mediapipe_hands, mediapipe_face = make_hands_detector()
    # Dictionaries for every hand.
    left_landmarks = {i: [] for i in range(0, 21)}
    right_landmarks = {i: [] for i in range(0, 21)}
    mouth_positions = []

    video_name = video_path.name
    video_data = read_video(video_path)

    frame_height = None
    frame_width = None

    for frame_i, frame in enumerate(video_data):
        # Detect hands in every frame of the video.
        if frame_height is None:
            frame_height, frame_width, _ = frame.shape

        frame = cv2.flip(frame, 1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Ony required when using the webcam, not recorded videos.
        frame.flags.writeable = False
        results = mediapipe_hands.process(frame)
        # print(results.multi_hand_landmarks)
        face_results = mediapipe_face.process(frame)
        frame.flags.writeable = True

        if not results.multi_hand_landmarks:
            # Nothing was detected this frame.
            for landmark_id in left_landmarks.keys():
                left_landmarks[landmark_id].append([-1.0, -1.0])
            for landmark_id in right_landmarks.keys():
                right_landmarks[landmark_id].append([-1.0, -1.0])
            # continue
        else:
            # Found some hands.
            found_left = False
            found_right = False
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_name = handedness.classification[0].label.lower()
                landmarks_coordinates = hand_landmarks.landmark
                if not gesture.uses_hand(hand_name):
                    # Detected a hand that should not have been tracked.
                    continue
                # if not is_landmark_in_active_zone(hand_landmarks.landmark):
                #     # Ignore when the hands are at the edge of the frame. The person is the video is still
                #     # getting in position and is not performing the gesture yet.
                #     continue
                landmark_list = calculate_landmark_list(
                    frame_width, frame_height, landmarks_coordinates
                )
                for landmark_id, landmark in enumerate(landmark_list):
                    if hand_name == "left":
                        left_landmarks[landmark_id].append(landmark)
                        # right_landmarks[landmark_id].append([-1, -1])
                        found_left = True
                    elif hand_name == "right":
                        right_landmarks[landmark_id].append(landmark)
                        # left_landmarks[landmark_id].append([-1, -1])
                        found_right = True

            # Get mouth:
            if not face_results.detections:
                if mouth_positions:
                    # Repeat the last position
                    mouth_positions.append(mouth_positions[-1])
            else:
                mouth = mp.solutions.face_detection.get_key_point(
                    face_results.detections[0],
                    mp.solutions.face_detection.FaceKeyPoint.MOUTH_CENTER,
                )
                mouth_positions.append([round(mouth.x, 2), round(mouth.y, 2)])

            if not found_left:
                for landmark_id in left_landmarks.keys():
                    left_landmarks[landmark_id].append([-1.0, -1.0])
            if not found_right:
                for landmark_id in right_landmarks.keys():
                    right_landmarks[landmark_id].append([-1.0, -1.0])

    return (
        video_name,
        (frame_width, frame_height),
        left_landmarks,
        right_landmarks,
        mouth_positions,
    )


class GestureDataset:
    def __init__(self, single_gesture=False):
        # self.X_dataset = []
        self.single_gesture = single_gesture
        self.gestures: List[GestureData] = []  # No data if loaded from csv file.

        self.x_data = np.array([])
        self.y_data = np.array([])
        self.lookup_dict = {}  # id -> gesture
        self.reverse_lookup_dict = {}  # gesture -> id

    def remove_gesture(self, gesture_name):
        """Removed a deleted gesture from the lookup dictionaries and the data arrays"""
        if gesture_name in self.reverse_lookup_dict:
            gesture_id = self.reverse_lookup_dict[gesture_name]
            delete_indexes = []
            for i, row in enumerate(self.y_data):
                if row == gesture_id:
                    delete_indexes.append(i)
            self.y_data = np.delete(self.y_data, delete_indexes, axis=0)
            self.x_data = np.delete(self.x_data, delete_indexes, axis=0)
            del self.lookup_dict[gesture_id]
            del self.reverse_lookup_dict[gesture_name]
        else:
            print(
                f"Warning: Could not delete {gesture_name} from dataset because it was not in the lookup keys."
            )

    def summary(self):
        print(f"Dataset contain {len(np.unique(self.y_data))} gestures.")

    def load_gestures_from_csv(self):
        for gesture in self.gestures:
            gesture_dataset = GestureDataset(single_gesture=True)
            gesture_dataset.gestures.append(gesture)
            gesture_dataset.load_from_csv(gesture.dataset_file)
            self.append_dataset(gesture_dataset)
        print("New lookup dict:")
        pprint(self.lookup_dict)

    def load_from_csv(self, csv_path: Path):
        if not csv_path.exists():
            raise Exception(f"Did not find the csv dataset at {csv_path}")
        X_dataset = []
        Y_dataset = []
        gesture_id = -1
        last_gesture_number = -1
        last_video_name = -1
        last_hand = -1

        gesture_videos_left_landmarks = {}
        gesture_videos_right_landmarks = {}

        dataset_left_x = []
        dataset_right_x = []
        dataset_left_y = []
        dataset_right_y = []
        dataset_left_angles = []
        dataset_right_angles = []

        with open(csv_path, "r", encoding="latin-1") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            for landmark_line in csv_reader:
                (
                    gesture_name,
                    gesture_number,
                    video_name,
                    frame_width,
                    frame_height,
                    hand_number,
                    landmark_id,
                    mouth_position,
                    history,
                ) = landmark_line
                frame_height = int(frame_height)
                frame_width = int(frame_width)
                landmark_id = int(landmark_id)
                gesture_number = int(gesture_number)
                video_name = video_name
                hand_number = int(hand_number)
                history = point_list_from_string(history)
                mouth_position = eval("lambda: " + mouth_position)()
                history = list(
                    map(
                        lambda coordinate: [float(coordinate[0]), float(coordinate[1])],
                        history,
                    )
                )

                # if ONLY_LANDMARK_ID and landmark_id != ONLY_LANDMARK_ID:
                # print("skipped csv", landmark_id)
                # continue

                if gesture_number != last_gesture_number:
                    last_gesture_number = gesture_number
                    gesture_id += 1
                    gesture_videos_left_landmarks[gesture_id] = {}
                    gesture_videos_right_landmarks[gesture_id] = {}
                    self.lookup_dict[gesture_id] = gesture_name
                    self.reverse_lookup_dict[gesture_name] = gesture_id

                if video_name != last_video_name:
                    last_video_name = video_name
                    gesture_videos_left_landmarks[gesture_id][video_name] = {
                        "landmarks": {i: [] for i in range(0, 21)},
                        "mouth_position": mouth_position,
                    }
                    gesture_videos_right_landmarks[gesture_id][video_name] = {
                        "landmarks": {i: [] for i in range(0, 21)},
                        "mouth_position": mouth_position,
                    }

                if last_hand != hand_number:
                    last_hand = hand_number

                if hand_number == 0:
                    gesture_videos_left_landmarks[gesture_id][video_name]["landmarks"][
                        landmark_id
                    ] = history
                elif hand_number == 1:
                    gesture_videos_right_landmarks[gesture_id][video_name]["landmarks"][
                        landmark_id
                    ] = history

        for gesture_id, videos in gesture_videos_left_landmarks.items():
            for video_name, video_data in videos.items():
                left_landmarks = video_data["landmarks"]
                try:
                    right_landmarks = gesture_videos_right_landmarks[gesture_id][
                        video_name
                    ]["landmarks"]
                    mouth_position = video_data["mouth_position"]
                    preprocess_landmarks(
                        left_landmarks, right_landmarks, frame_width, frame_height
                    )
                    left_angles = process_orientation(left_landmarks)
                    right_angles = process_orientation(right_landmarks)
                    # left_angles = make_list_fixed_length(
                    #     left_angles,
                    #     MAX_VIDEO_FRAMES,
                    #     combine_function=lambda existing, value: (existing + value) / 2,
                    #     create_function=lambda prev_value, next_value: (
                    #         prev_value + next_value
                    #     )
                    #     / 2,
                    # )
                    # right_angles = make_list_fixed_length(
                    #     right_angles,
                    #     MAX_VIDEO_FRAMES,
                    #     combine_function=lambda existing, value: (existing + value) / 2,
                    #     create_function=lambda prev_value, next_value: (
                    #         prev_value + next_value
                    #     )
                    #     / 2,
                    # )

                    left_hand_openness = hand_openness(landmarks=left_landmarks)
                    right_hand_openness = hand_openness(landmarks=right_landmarks)

                    for i, landmarks in left_landmarks.items():
                        # if ONLY_LANDMARK_ID and i != ONLY_LANDMARK_ID:
                        #     continue
                        left_landmarks[i] = pre_process_point_history_mouth_position(
                            mouth_position, landmarks
                        )
                    for i, landmarks in right_landmarks.items():
                        # if ONLY_LANDMARK_ID and i != ONLY_LANDMARK_ID:
                        #     continue
                        right_landmarks[i] = pre_process_point_history_mouth_position(
                            mouth_position, landmarks
                        )

                    x_data = [
                        left_landmarks[ONLY_LANDMARK_ID][0],
                        left_landmarks[ONLY_LANDMARK_ID][1],
                        right_landmarks[ONLY_LANDMARK_ID][0],
                        right_landmarks[ONLY_LANDMARK_ID][1],
                        left_angles,
                        right_angles,
                        *left_hand_openness,
                        *right_hand_openness
                    ]
                    Y_dataset.append(gesture_id)
                    X_dataset.append(x_data)
                except IndexError as e:
                    print(
                        f"Something went wrong while processing {self.lookup_dict[gesture_id]}/{video_name}: {e}"
                    )
                    # raise e
        # self.X_dataset = copy.deepcopy(X_dataset)

        self.y_data = np.array(Y_dataset)
        self.x_data = np.array(X_dataset, dtype=np.float)

    def append_dataset(self, other_dataset: "GestureDataset"):
        """Adds a dataset to this dataset."""
        if len(other_dataset.y_data) == 0:
            return
        new_y_data = other_dataset.y_data
        new_x_data = other_dataset.x_data
        if len(self.y_data) > 0:
            gesture_id_base = np.amax(self.y_data) + 1
            new_y_data = new_y_data + gesture_id_base
            self.x_data = np.concatenate([self.x_data, new_x_data])
            self.y_data = np.concatenate([self.y_data, new_y_data])
        else:
            gesture_id_base = 0
            self.x_data = other_dataset.x_data
            self.y_data = other_dataset.y_data
        for id, gesture_name in other_dataset.lookup_dict.items():
            self.lookup_dict[id + gesture_id_base] = gesture_name
            self.reverse_lookup_dict[gesture_name] = id + gesture_id_base
        # self.X_dataset += other_dataset.X_dataset

    def update_gesture_dataset(self, other_dataset: "GestureDataset"):
        """Adds a dataset to this dataset."""
        if len(other_dataset.y_data) == 0:
            return
        new_y_data = other_dataset.y_data
        new_x_data = other_dataset.x_data
        gesture_id = 0
        gesture_name = None
        for gesture in other_dataset.gestures:
            gesture_id = self.reverse_lookup_dict[gesture.name]
            gesture_name = gesture.name
            self.remove_gesture(gesture.name)
        new_y_data.fill(gesture_id)
        print(
            f"Updated dataset. Used gesture_id={gesture_id} and new_y_data={new_y_data}"
        )
        self.x_data = np.concatenate([self.x_data, new_x_data])
        self.y_data = np.concatenate([self.y_data, new_y_data])
        self.lookup_dict[gesture_id] = gesture_name
        self.reverse_lookup_dict[gesture_name] = gesture_id
        pprint(self.y_data)
        pprint(self.lookup_dict)

    def analyze_videos(self, csv_out_path: Optional[Path] = None, overwrite=True):
        """
        Use mediapipe to detect hand landmarks in every training video and save this data in a usable format.
        """
        print(
            f"Analysing {sum([len(g.reference_videos) for g in self.gestures])} video files."
        )
        if csv_out_path and overwrite and csv_out_path.exists():
            os.remove(csv_out_path)
        start_time = time.time()
        for gesture_i, gesture in enumerate(self.gestures):
            # Loop over all gestures in the dataset.
            with Pool(processes=8) as pool:
                # Executes the detect_hands_task in parallel.
                results = pool.imap_unordered(
                    functools.partial(detect_hands_task, gesture),
                    gesture.reference_videos,
                )
                with open(csv_out_path, "a", newline="") as f:
                    writer = csv.writer(f, delimiter=";")
                    for video_i, result in enumerate(results):
                        (
                            video_name,
                            (frame_width, frame_height),
                            left_landmarks,
                            right_landmarks,
                            mouth_positions,
                        ) = result
                        if mouth_positions:
                            reference_mouth_position = mouth_positions[
                                int(len(mouth_positions) / 2)
                            ]
                        else:
                            print(
                                f"Warning: No mouth detected in {gesture.name}/{video_name}. Using default value"
                            )
                            reference_mouth_position = [0.5, 0.4]
                        try:
                            # preprocess_landmarks(left_landmarks, right_landmarks, frame_width, frame_height)
                            for landmark_id, values in left_landmarks.items():
                                log_csv(
                                    writer,
                                    gesture_name=gesture.name,
                                    gesture_number=gesture_i,
                                    video_number=video_name,
                                    hand_number=0,
                                    landmark_id=landmark_id,
                                    mouth_position=reference_mouth_position,
                                    point_history_list=values,
                                    width=frame_width,
                                    height=frame_height,
                                )
                            for landmark_id, values in right_landmarks.items():
                                log_csv(
                                    writer,
                                    gesture_name=gesture.name,
                                    gesture_number=gesture_i,
                                    video_number=video_name,
                                    hand_number=1,
                                    landmark_id=landmark_id,
                                    mouth_position=reference_mouth_position,
                                    point_history_list=values,
                                    width=frame_width,
                                    height=frame_height,
                                )
                        except Exception as e:
                            print(f"Failure on a video for gesture {gesture.name}: {e}")
                            raise e
        print(f"Completed analyzing videos in {time.time() - start_time}")

    def scan_videos(self, dataset_location: Path, handedness_data):
        # TODO: Refactor this. Some duplicate code.
        self.gestures.clear()
        if not self.single_gesture:
            gesture_folders = clean_listdir(dataset_location)
        else:
            gesture_folders = [dataset_location]
        for gesture_folder in gesture_folders:
            if not self.single_gesture:
                gesture_name = gesture_folder.split("_")[0]
                gesture_path = dataset_location / str(gesture_folder)
            else:
                gesture_name = gesture_folder.name.split("_")[0]
                gesture_path = dataset_location
            videos = clean_listdir(gesture_path)
            left_hand, right_hand = handedness_data[gesture_name]
            gesture = GestureData(
                name=gesture_name,
                left_hand=left_hand,
                right_hand=right_hand,
                dataset_file=dataset_location / gesture_folder / "dataset.csv",
            )
            for video_name in videos:
                video_name = str(video_name)
                if is_video(video_name):
                    gesture.add_video(gesture_path / str(video_name))
            self.gestures.append(gesture)

    def __len__(self):
        return len(np.unique(self.y_data))


if __name__ == "__main__":
    CSV_OUT_PATH = Path("gestures_dataset.csv")
    DATASET_LOCATION = Path("ai_data/vgt-all")
    # dataset = GestureDataset(single_gesture=False)
    # handedness_data = {}
    only_gestures = []
    ONLY_NEW = True
    for gesture_folder in tqdm(clean_listdir(DATASET_LOCATION)):
        # print(f"Creating dataset.csv for {gesture_folder}")
        gesture_folder_path = DATASET_LOCATION / gesture_folder
        *gestures_name_parts, handedness_string = gesture_folder.split("_")
        gesture_name = "_".join(gestures_name_parts)
        if only_gestures and gesture_name.lower() not in only_gestures:
            continue
        if ONLY_NEW and "dataset.csv" in clean_listdir(gesture_folder_path):
            continue
        new_gesture_dataset = GestureDataset(single_gesture=True)
        new_gesture_dataset.scan_videos(
            gesture_folder_path,
            handedness_data={
                gesture_name: (handedness_string[0] == "1", handedness_string[1] == "1")
            },
        )
        print(f"Gesture: {gesture_name}")
        new_gesture_dataset.analyze_videos(
            csv_out_path=gesture_folder_path / "dataset.csv"
        )
        # new_gesture_dataset.load_from_csv(gesture_folder_path / "dataset.csv")
