import copy
import csv
import functools
import itertools
import os
import time
from math import sqrt
from pathlib import Path
from pprint import pprint
from typing import List, Union, Optional, Tuple, Dict, Any

import cv2
import mediapipe as mp
import numpy as np
import skvideo.io
from matplotlib import pyplot as plt
from tqdm import tqdm

from sl_ai.config import MAX_VIDEO_FRAMES, ONLY_LANDMARK_ID
from multiprocessing import Pool

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
            *[p for p in point_history_list],
        ]
    )


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


def shrink_coordinates_list(
    lst: List[Tuple[float]], new_size: int
) -> List[Tuple[float]]:
    """
    Shrink a list of numbers (coordinates) to a fixed length.
    When shrinking a list some values will need to be discarded because they will want to end up at the same location.
    Instead of completely discarding these values, insert a number inbetween the 2 numbers when they would end up at the same index.
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
            existing_x, existing_y = existing_at_index
            x, y = value
            result[insert_index] = [(existing_x + x) / 2, (existing_y + y) / 2]
        else:
            result[insert_index] = value
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
    Uses extend_coordinates_list and shrink_coordinates_list to resize a list to a specific size.
    """
    lst_len = len(lst)
    if lst_len == new_size:
        return copy.deepcopy(lst)
    elif lst_len > new_size:
        return shrink_coordinates_list(lst, new_size)
    else:
        return extend_coordinates_list(lst, new_size)


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


def mirror_landmarks_list(landmarks: List[Tuple[float]]) -> List[Tuple[float]]:
    """
    Mirrors the x coordinates of all landmarks.
    Data coming in from the webcam/browser is mirrored and will not match the dataset videos.
    """
    mirrored_landmarks = []
    for _, landmark in enumerate(landmarks):
        mirrored_landmarks.append([1 - landmark[0], landmark[1]])
    return mirrored_landmarks


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
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_point_history_center(
    image_width: int, image_height: int, point_history: List[Tuple[float]]
) -> List[float]:
    """Alternative data representation for the AI model. Converts a list of coordinates to a list of distances to the center of the screen."""
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0.5, 0.5
    for index, point in enumerate(temp_point_history):
        if point == [-1, -1]:
            continue
        temp_point_history[index][0] = temp_point_history[index][0] - base_x
        temp_point_history[index][1] = temp_point_history[index][1] - base_y
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history)
    )  # [[x, y], [x, y]] => [x, y, x, y] # TODO: Make this a separate function.
    return temp_point_history


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
    trim_landmark_lists(left_landmarks, right_landmarks, [-1, -1])
    for landmark_id, landmarks in left_landmarks.items():
        landmarks = make_coordinates_list_fixed_length(landmarks, MAX_VIDEO_FRAMES)
        landmarks = fill_holes(landmarks, [-1, -1])
        # landmarks = pre_process_point_history(frame_width, frame_height, landmarks)
        left_landmarks[landmark_id] = landmarks

    for landmark_id, landmarks in right_landmarks.items():
        landmarks = make_coordinates_list_fixed_length(landmarks, MAX_VIDEO_FRAMES)
        landmarks = fill_holes(landmarks, [-1, -1])
        # landmarks = pre_process_point_history(frame_width, frame_height, landmarks)
        right_landmarks[landmark_id] = landmarks


def make_hands_detector() -> mp.solutions.hands.Hands:
    """Creates and instance of the mediapipe hands detector."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        model_complexity=1,
    )
    return hands


class GestureData:
    # TODO: This class might not be required and is confusing.
    def __init__(self, name: str, left_hand: bool = True, right_hand: bool = True):
        self.name: str = name
        self.left_hand: bool = left_hand
        self.right_hand: bool = right_hand
        self.reference_videos: List[Path] = []

    def add_video(self, video_path: Path):
        self.reference_videos.append(video_path)

    def uses_hand(self, hand_name: str) -> bool:
        if hand_name.lower() == "left":
            return self.left_hand
        elif hand_name.lower() == "right":
            return self.right_hand
        return False

    def __str__(self):
        return f"Gesture: {self.name}"


def detect_hands_task(gesture: GestureData, video_path: Path):
    """Detects gesture landmarks from a video file."""
    mediapipe_hands = make_hands_detector()
    frame_height = None
    frame_width = None
    # Dictionaries for every hand.
    left_landmarks = {i: [] for i in range(0, 21)}
    right_landmarks = {i: [] for i in range(0, 21)}

    video_name = video_path.name
    video_data = read_video(video_path)

    frame_height = None
    frame_width = None

    for frame_i, frame in enumerate(video_data):
        # Detect hands in every frame of the video.
        if frame_height is None:
            frame_height, frame_width, _ = frame.shape

        # frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        results = mediapipe_hands.process(frame)

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
            if not found_left:
                for landmark_id in left_landmarks.keys():
                    left_landmarks[landmark_id].append([-1, -1])
            if not found_right:
                for landmark_id in right_landmarks.keys():
                    right_landmarks[landmark_id].append([-1, -1])

    return video_name, (frame_width, frame_height), left_landmarks, right_landmarks


class GestureDataset:
    def __init__(self, single_gesture=False):
        self.single_gesture = single_gesture
        self.gestures: List[GestureData] = []  # No data if loaded from csv file.

        self.x_data = None
        self.y_data = None
        self.lookup_dict = {}
        self.reverse_lookup_dict = {}

    def remove_gesture(self, gesture_name):
        """Removed a deleted gesture from the lookup dictionaries and the data arrays"""
        if gesture_name in self.reverse_lookup_dict:
            gesture_id = self.reverse_lookup_dict[gesture_name]
            delete_indexes = []
            for i, row in enumerate(self.y_data):
                if row == gesture_id:
                    delete_indexes.append(i)
            print(self.y_data.shape, self.x_data.shape)
            self.y_data = np.delete(self.y_data, delete_indexes, axis=0)
            self.x_data = np.delete(self.x_data, delete_indexes, axis=0)
            print("Deleted:", delete_indexes)
            print(self.y_data.shape, self.x_data.shape)
            del self.lookup_dict[gesture_id]
            del self.reverse_lookup_dict[gesture_name]
        else:
            print(
                f"Warning: Could not delete {gesture_name} from dataset because it was not in the lookup keys."
            )

    def summary(self):
        print(f"Dataset contain {len(np.unique(self.y_data))} gestures.")

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
                    *history,
                ) = landmark_line
                frame_height = int(frame_height)
                frame_width = int(frame_width)
                landmark_id = int(landmark_id)
                gesture_number = int(gesture_number)
                video_name = video_name
                hand_number = int(hand_number)

                history = list(map(lambda coordinate: eval(coordinate), history))
                history = list(
                    map(
                        lambda coordinate: [float(coordinate[0]), float(coordinate[1])],
                        history,
                    )
                )

                if gesture_number != last_gesture_number:
                    last_gesture_number = gesture_number
                    gesture_id += 1
                    gesture_videos_left_landmarks[gesture_id] = {}
                    gesture_videos_right_landmarks[gesture_id] = {}
                    print(f"{gesture_name}({gesture_number} -> {gesture_id})")
                    self.lookup_dict[gesture_id] = gesture_name
                    self.reverse_lookup_dict[gesture_name] = gesture_id

                if video_name != last_video_name:
                    last_video_name = video_name
                    gesture_videos_left_landmarks[gesture_id][video_name] = {
                        i: [] for i in range(0, 21)
                    }
                    gesture_videos_right_landmarks[gesture_id][video_name] = {
                        i: [] for i in range(0, 21)
                    }

                if last_hand != hand_number:
                    last_hand = hand_number
                    # plt.close()
                # if landmark_id != ONLY_LANDMARK_ID:
                #     gesture_videos_left_landmarks[gesture_id][video_name][landmark_id] = [[0, 0] for _ in range(10)]
                #     gesture_videos_right_landmarks[gesture_id][video_name][landmark_id] = [[0, 0] for _ in range(10)]

                if hand_number == 0:
                    gesture_videos_left_landmarks[gesture_id][video_name][
                        landmark_id
                    ] = history
                elif hand_number == 1:
                    gesture_videos_right_landmarks[gesture_id][video_name][
                        landmark_id
                    ] = history

        for gesture_id, videos in gesture_videos_left_landmarks.items():
            for video_id, left_landmarks in videos.items():
                try:
                    right_landmarks = gesture_videos_right_landmarks[gesture_id][
                        video_id
                    ]
                    preprocess_landmarks(
                        left_landmarks, right_landmarks, frame_width, frame_height
                    )
                    for i, landmarks in left_landmarks.items():
                        if ONLY_LANDMARK_ID and i != ONLY_LANDMARK_ID:
                            continue
                        left_landmarks[i] = pre_process_point_history_center(
                            None, None, landmarks
                        )
                    for i, landmarks in right_landmarks.items():
                        if ONLY_LANDMARK_ID and i != ONLY_LANDMARK_ID:
                            continue
                        right_landmarks[i] = pre_process_point_history_center(
                            None, None, landmarks
                        )
                    Y_dataset.append(gesture_id)
                    X_dataset.append(
                        left_landmarks[ONLY_LANDMARK_ID]
                        + right_landmarks[ONLY_LANDMARK_ID]
                    )
                except IndexError as e:
                    print(f"Something went wrong while processing {video_name}: {e}")
                except Exception as e:
                    print(f"Something went wrong while processing {video_name}: {e}")

        self.y_data = np.array(Y_dataset)
        self.x_data = np.array(X_dataset, dtype="float32")

    def append_dataset(self, other_dataset: "GestureDataset"):
        if len(other_dataset.y_data) == 0:
            return
        gesture_id_base = np.amax(self.y_data) + 1
        new_y_data = other_dataset.y_data + gesture_id_base
        new_x_data = other_dataset.x_data
        for id, gesture_name in other_dataset.lookup_dict.items():
            self.lookup_dict[id + gesture_id_base] = gesture_name
            self.reverse_lookup_dict[gesture_name] = id + gesture_id_base
        self.y_data = np.concatenate([self.y_data, new_y_data])
        self.x_data = np.concatenate([self.x_data, new_x_data])

    def analyze_videos(self, csv_out_path: Optional[Path] = None, overwrite=False):
        """
        Use mediapipe to detect hand landmarks in every training video and save this data in a usable format.
        """
        print(
            f"Analysing video {sum([len(g.reference_videos) for g in self.gestures])} files."
        )
        if csv_out_path and overwrite and csv_out_path.exists():
            os.remove(csv_out_path)
        print(csv_out_path)
        start_time = time.time()
        for gesture_i, gesture in enumerate(tqdm(self.gestures)):
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
                        ) = result
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
                                    point_history_list=values,
                                    width=frame_width,
                                    height=frame_height,
                                )
                        except Exception as e:
                            print(f"Failure on a video for gesture {gesture.name}: {e}")
        print(f"Completed analyzing videos in {time.time() - start_time}")

    def scan_videos(self, dataset_location: Path, handedness_data):
        # TODO: Refactor this. Some duplicate code.
        self.gestures.clear()
        if not self.single_gesture:
            gesture_folders = os.listdir(dataset_location)
        else:
            gesture_folders = [dataset_location]
        for gesture_folder in gesture_folders:
            if not self.single_gesture:
                gesture_name = gesture_folder.split("_")[0]
                gesture_path = dataset_location / str(gesture_folder)
            else:
                gesture_name = gesture_folder.name.split("_")[0]
                gesture_path = dataset_location

            left_hand, right_hand = list(handedness_data.values())[0]
            gesture = GestureData(
                name=gesture_name, left_hand=left_hand, right_hand=right_hand
            )
            for video_name in os.listdir(gesture_path):
                video_name = str(video_name)
                if (
                    video_name.endswith("mp4")
                    or video_name.endswith("mkv")
                    or video_name.endswith("MOV")
                ):
                    gesture.add_video(gesture_path / str(video_name))
            self.gestures.append(gesture)

        print(f"Loaded {len(self.gestures)} gestures")

    def add_django_gesture(self, django_gesture: "Gesture"):
        gesture_data = GestureData(
            name=django_gesture.word,
            left_hand=django_gesture.left_hand,
            right_hand=django_gesture.right_hand,
        )
        gesture_path = Path("sl_ai/ai_data/vgt-uploaded")
        if django_gesture.creator:
            gesture_path = gesture_path / str(django_gesture.creator.id)
        gesture_path = gesture_path / django_gesture.handed_string
        for video_name in os.listdir(gesture_path):
            gesture_data.add_video(gesture_path / str(video_name))
        self.gestures.append(gesture_data)

    def __len__(self):
        return len(np.unique(self.y_data))


if __name__ == "__main__":
    CSV_OUT_PATH = Path("gestures_dataset.csv")
    DATASET_LOCATION = Path("ai_data/vgt-all")

    dataset = GestureDataset(single_gesture=False)
    handedness_data = {}
    for gesture_folder in os.listdir(DATASET_LOCATION):
        gesture_folder_path = DATASET_LOCATION / gesture_folder
        gesture_name, handedness_string = gesture_folder.split("_")
        handedness_data[gesture_name] = (
            handedness_string[0] == "1",
            handedness_string[1] == "1",
        )
    dataset.scan_videos(
        dataset_location=DATASET_LOCATION, handedness_data=handedness_data
    )
    dataset.analyze_videos(csv_out_path=CSV_OUT_PATH, overwrite=True)
    dataset.load_from_csv(CSV_OUT_PATH)
