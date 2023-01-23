import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from django.db.models import Q
from django.http import JsonResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status, viewsets, permissions
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication, TokenAuthentication
from pprint import pprint

from sign_language_app import serializers
from sign_language_app.classifier import Classifier
from sign_language_app.models import Gesture, Unit, UnitAttempt, GestureAttempt
from sign_language_app.utils import get_user, is_admin
from sl_ai.dataset import preprocess_landmarks, trim_landmark_lists, calculate_landmark_list, \
    pre_process_point_history_center
from sign_language_app.background_tasks import retrain_model

FRAME_WIDTH = 720
FRAME_HEIGHT = 480


@api_view(['GET', 'POST', 'OPTIONS'])
@authentication_classes([SessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def trigger_retrain_model(request):
    user = get_user(request)
    if is_admin(user):
        retrain_model()
        return JsonResponse({'status': 'OK'}, status=status.HTTP_201_CREATED)
    return JsonResponse({'status': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)



@api_view(['POST', 'OPTIONS'])
@authentication_classes([SessionAuthentication, BasicAuthentication])
# @permission_classes([IsAuthenticated])
# @authentication_classes([])
@permission_classes([])
def test_auth(request):
    data = request.data
    hand_frames = data['hand_frames']
    # gesture_id = data['gesture_id']
    print(f"Gesture ID: {data['gesture']['id']}")
    print(f'Frames: {len(hand_frames)}')
    gesture = get_object_or_404(Gesture, pk=int(data['gesture']['id']))
    # print(left)

    left_landmarks = {i: [] for i in range(0, 21)}
    right_landmarks = {i: [] for i in range(0, 21)}

    # TODO: Refactor this.
    # TODO: Check if is_landmark_in_active_zone
    for frame in hand_frames:
        left, right = frame
        if left:# and gesture.left_hand:
            landmark_list_left = calculate_landmark_list(FRAME_WIDTH, FRAME_HEIGHT, left)
            for landmark_id, landmark in enumerate(landmark_list_left):
                left_landmarks[landmark_id].append(landmark)
        else:
            for landmark_id in left_landmarks.keys():
                left_landmarks[landmark_id].append([-1, -1])

        if right:# and gesture.right_hand:
            landmark_list_right = calculate_landmark_list(FRAME_WIDTH, FRAME_HEIGHT, right)
            for landmark_id, landmark in enumerate(landmark_list_right):
                right_landmarks[landmark_id].append(landmark)
        else:
            for landmark_id in right_landmarks.keys():
                right_landmarks[landmark_id].append([-1, -1])

    if left_landmarks == right_landmarks:
        is_correct = 0
        print("WARNING: Nothing was detected.")

    else:
        # TODO: Verify if the js and python coordinate systems are the same.
        preprocess_landmarks(left_landmarks, right_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
        for i, landmarks in left_landmarks.items():
            left_landmarks[i] = pre_process_point_history_center(None, None, landmarks)
        for i, landmarks in right_landmarks.items():
            right_landmarks[i] = pre_process_point_history_center(None, None, landmarks)
        # left_landmarks = pre_process_point_history_center(None, None, left_landmarks)
        # right_landmarks = pre_process_point_history_center(None, None, right_landmarks)
        result = Classifier().gesture_classifier.predict(left_landmarks, right_landmarks)
        classes_x = np.argmax(result, axis=1)
        print(classes_x)
        prediction_percents = (result*100)
        frame = pd.DataFrame(prediction_percents.astype(np.uint8))
        print(frame)
        is_correct = random.randint(0, 1) == 1
    return JsonResponse({'status': 'OK', "correct": 0}, status=status.HTTP_201_CREATED)


class GestureViewSet(viewsets.ModelViewSet):
    serializer_class = serializers.GestureSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        qs = Gesture.objects
        word_like = self.request.query_params.get("name_like")
        if word_like:
            qs = qs.filter(Q(word__icontains=word_like))
        return qs.all()
