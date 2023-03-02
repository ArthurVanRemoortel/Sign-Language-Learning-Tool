import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from django.db.models import Q, Avg
from django.http import JsonResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from matplotlib import pyplot as plt
from rest_framework import status, viewsets, permissions
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import (
    SessionAuthentication,
    BasicAuthentication,
    TokenAuthentication,
)
from pprint import pprint

from sign_language_app import serializers
from sign_language_app.classifier import Classifier
from sign_language_app.models import Gesture, Unit, UnitAttempt, GestureAttempt
from sign_language_app.utils import get_user, is_admin
from sl_ai.config import ONLY_LANDMARK_ID
from sl_ai.dataset import (
    preprocess_landmarks,
    trim_landmark_lists,
    calculate_landmark_list,
    pre_process_point_history_center,
    mirror_landmarks_list,
    mirror_coordinate,
    pre_process_point_history_mouth_position,
    process_orientation,
    hand_openness,
)
from sign_language_app.background_tasks import retrain_model


def visualize_gesture(coordinates, frame_width, frame_height):
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
    plt.show()


@api_view(["GET", "POST", "OPTIONS"])
@authentication_classes([SessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def trigger_retrain_model(request):
    user = get_user(request)
    if is_admin(user):
        retrain_model()
        return JsonResponse({"status": "OK"}, status=status.HTTP_201_CREATED)
    return JsonResponse({"status": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)


@api_view(["POST", "OPTIONS"])
@authentication_classes([SessionAuthentication, BasicAuthentication])
# @permission_classes([IsAuthenticated])
# @authentication_classes([])
@permission_classes([])
def verify_gesture(request):
    data = request.data
    hand_frames = data["hand_frames"]
    # gesture_id = data['gesture_id']
    print(f"Gesture ID: {data['gesture']['id']}")
    print(f"Frames: {len(hand_frames)}")
    gesture: Gesture = get_object_or_404(Gesture, pk=int(data["gesture"]["id"]))
    frame_width = data["frame_width"]
    frame_height = data["frame_height"]

    # print(left)

    left_landmarks = {i: [] for i in range(0, 21)}
    right_landmarks = {i: [] for i in range(0, 21)}
    mouth: [float, float] = []
    # TODO: Refactor this.
    # TODO: Check if is_landmark_in_active_zone
    for frame in hand_frames:
        left, right, mouth = frame
        mouth = mirror_coordinate(mouth["x"], mouth["y"])
        # print("mouth", mouth)
        if left and gesture.left_hand:
            landmark_list_left = calculate_landmark_list(
                frame_width, frame_height, left
            )
            landmark_list_left = mirror_landmarks_list(landmark_list_left)
            for landmark_id, landmark in enumerate(landmark_list_left):
                left_landmarks[landmark_id].append(landmark)
        else:
            for landmark_id in left_landmarks.keys():
                left_landmarks[landmark_id].append([-1.0, -1.0])

        if right and gesture.right_hand:
            landmark_list_right = calculate_landmark_list(
                frame_width, frame_height, right
            )
            landmark_list_right = mirror_landmarks_list(landmark_list_right)
            for landmark_id, landmark in enumerate(landmark_list_right):
                right_landmarks[landmark_id].append(landmark)
        else:
            for landmark_id in right_landmarks.keys():
                right_landmarks[landmark_id].append([-1.0, -1.0])

    # visualize_gesture(
    #     frame_width=frame_width,
    #     frame_height=frame_height,
    #     coordinates=right_landmarks[ONLY_LANDMARK_ID],
    # )
    print(
        "left: ",
        len(
            list(
                filter(lambda p: p != [-1.0, -1.0], (left_landmarks[ONLY_LANDMARK_ID]))
            )
        ),
    )
    print(
        "right: ",
        len(
            list(
                filter(lambda p: p != [-1.0, -1.0], (right_landmarks[ONLY_LANDMARK_ID]))
            )
        ),
    )
    if left_landmarks == right_landmarks:
        is_correct = 0
        print("WARNING: Nothing was detected.")
    else:
        preprocess_landmarks(left_landmarks, right_landmarks, frame_width, frame_height)

        left_angles = process_orientation(left_landmarks)
        right_angles = process_orientation(right_landmarks)
        left_hand_openness = hand_openness(landmarks=left_landmarks)
        right_hand_openness = hand_openness(landmarks=right_landmarks)

        for i, landmarks in left_landmarks.items():
            left_landmarks[i] = pre_process_point_history_mouth_position(
                mouth, landmarks
            )
        for i, landmarks in right_landmarks.items():
            right_landmarks[i] = pre_process_point_history_mouth_position(
                mouth, landmarks
            )

        result = Classifier().gesture_classifier.predict(
            left_x=left_landmarks[ONLY_LANDMARK_ID][0],
            left_y=left_landmarks[ONLY_LANDMARK_ID][1],
            right_x=right_landmarks[ONLY_LANDMARK_ID][0],
            right_y=right_landmarks[ONLY_LANDMARK_ID][1],
            left_angles=left_angles,
            right_angles=right_angles,
            left_openness=left_hand_openness,
            right_openness=right_hand_openness,
        )
        print(f"Predictions for {gesture}")
        predicted_gestures = {}
        for gesture_id, prediction in enumerate(result[0]):
            prediction = int(prediction * 100)
            if prediction > 5:
                try:
                    predicted_gestures[
                        Classifier().gesture_classifier.gesture_dataset.lookup_dict[
                            gesture_id
                        ]
                    ] = prediction
                except KeyError:
                    print(
                        f"Warning: {gesture_id} is present in the model but was probably deleted from the dataset. Retrain the model."
                    )
            elif (
                prediction > 1
                and gesture_id
                in Classifier().gesture_classifier.gesture_dataset.lookup_dict
            ):
                print(
                    f"Other prediction but confidence was low: {Classifier().gesture_classifier.gesture_dataset.lookup_dict[gesture_id]} = {prediction}"
                )
        is_correct = (
            gesture.word.lower() in predicted_gestures
            or gesture.word in predicted_gestures
        )
        print(f"Best prediction: {predicted_gestures}, is_correct={is_correct}")
    return JsonResponse(
        {"status": "OK", "correct": is_correct}, status=status.HTTP_201_CREATED
    )


class GestureViewSet(viewsets.ModelViewSet):
    serializer_class = serializers.GestureSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        qs = Gesture.objects
        word_like = self.request.query_params.get("name_like")
        if word_like:
            qs = qs.filter(Q(word__icontains=word_like))
        return qs.all()


class UnitAttemptsViewSet(viewsets.ModelViewSet):
    serializer_class = serializers.UnitAttemptsSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        qs = UnitAttempt.objects.order_by("datetime")
        unit_id = self.request.query_params.get("unit")
        user_id = self.request.query_params.get("user")
        if unit_id:
            qs = qs.filter(Q(unit=unit_id))
        if user_id:
            qs = qs.filter(Q(user=user_id))
        return qs.all()
