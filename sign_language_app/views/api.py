import os
import random
from pathlib import Path

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from pprint import pprint
from sign_language_app.classifier import classifier
from sl_ai.dataset import preprocess_landmarks, trim_landmark_lists, calculate_landmark_list

FRAME_WIDTH = 720
FRAME_HEIGHT = 480


@api_view(['POST', 'OPTIONS'])
@authentication_classes([SessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def test_auth(request):
    data = request.data
    hand_frames = data['hand_frames']

    # print(left)

    print(f"Gesture: {data['gesture_id']}")
    print(f'Left: {len(hand_frames)} frames.')

    left_landmarks = {i: [] for i in range(0, 21)}
    right_landmarks = {i: [] for i in range(0, 21)}

    # TODO: Refactor this.
    for frame in hand_frames:
        left, right = frame
        if left:
            landmark_list_left = calculate_landmark_list(FRAME_WIDTH, FRAME_HEIGHT, left)
            for landmark_id, landmark in enumerate(landmark_list_left):
                left_landmarks[landmark_id].append(landmark)
        else:
            for landmark_id in left_landmarks.keys():
                left_landmarks[landmark_id].append([-1, -1])

        if right:
            landmark_list_right = calculate_landmark_list(FRAME_WIDTH, FRAME_HEIGHT, right)
            for landmark_id, landmark in enumerate(landmark_list_right):
                right_landmarks[landmark_id].append(landmark)
        else:
            for landmark_id in right_landmarks.keys():
                right_landmarks[landmark_id].append([-1, -1])


    preprocess_landmarks(left_landmarks, right_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
    result = classifier.predict(left_landmarks, right_landmarks)
    classes_x = np.argmax(result, axis=1)
    print(classes_x)

    is_correct = random.randint(0, 1) == 1
    return JsonResponse({'status': 'OK', "correct": is_correct}, status=status.HTTP_201_CREATED)
