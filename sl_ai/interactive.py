from pprint import pprint

import cv2
import mediapipe as mp

try:
    from sl_ai.dataset import process_orientation, calculate_landmark_list, hand_openness, calculate_distance
except:
    from dataset import process_orientation, calculate_landmark_list, hand_openness, calculate_distance

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection


def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i >= 0:
        print(f"Trying camera {i}")
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

# pprint(returnCameraIndexes())
# index = 0
# arr = []
# while True:
#     cap = cv2.VideoCapture(index)
#     if not cap.read()[0]:
#         break
#     else:
#         arr.append(index)
#     cap.release()
#     index += 1
# print(arr)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5
) as hands:
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            face_results = face.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            left_ear: [float, float] = None
            right_ear: [float, float] = None

            if face_results.detections:
                for detection in face_results.detections:
                    mp_drawing.draw_detection(image, detection)
                    left_ear_temp = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EAR_TRAGION)
                    left_ear = [left_ear_temp.x, left_ear_temp.y]
                    right_ear_temp = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.RIGHT_EAR_TRAGION)
                    right_ear = [right_ear_temp.x, right_ear_temp.y]

            if results.multi_hand_landmarks and left_ear and right_ear:
                left_angle = None
                right_angle = None
                left_hand_openness = None
                right_hand_openness = None
                for hand_landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks, results.multi_handedness
                    ):
                        hand_name = handedness.classification[0].label.lower()
                        landmarks_coordinates = hand_landmarks.landmark
                        landmark_list = calculate_landmark_list(
                            None, None, landmarks_coordinates
                        )
                        lst = {landmark_id: [landmark] for landmark_id, landmark in enumerate(landmark_list)}
                        # print(hand_name, lst)
                        if hand_name == "left":
                            left_angle = process_orientation(landmarks=lst)
                        if hand_name == "right":
                            right_angle = process_orientation(landmarks=lst)

                        if hand_name == "left":
                            left_hand_openness = hand_openness(landmarks=lst)
                        if hand_name == "right":
                            right_hand_openness = hand_openness(landmarks=lst)



                print(left_hand_openness, right_hand_openness)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
cap.release()
