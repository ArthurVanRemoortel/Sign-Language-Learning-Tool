import os
import random
import threading
import time
from pathlib import Path
from pprint import pprint
from typing import List

import schedule
from django.db import transaction
from django.db.models import Q

from learning_site.settings import (
    BACKGROUND_TRAINING_TIME,
    UPLOADED_GESTURES_ROOT,
    SAVED_MODEL_PATH,
)
from sign_language_app.classifier import Classifier
from sign_language_app.models import Gesture
from sl_ai.dataset import GestureDataset

IS_TRAINING = False


def retrain_thread(new_gestures: List[Gesture], deleted_gestures: List[Gesture]):
    global IS_TRAINING
    if not new_gestures and not deleted_gestures:
        return
    try:
        with transaction.atomic():
            IS_TRAINING = True
            print("Starting training...")
            if deleted_gestures:
                for deleted_gesture in deleted_gestures:
                    Classifier().gesture_classifier.gesture_dataset.remove_gesture(
                        gesture_name=deleted_gesture.word
                    )
                    deleted_gesture.delete()

            for gesture in new_gestures:
                gesture.status = Gesture.Status.TRAINING
                gesture.save()

            for gesture in new_gestures:
                print(f"Training gesture: {gesture}")
                gesture_dataset = GestureDataset(single_gesture=True)
                gesture_dataset.scan_videos(
                    gesture.videos_location,
                    handedness_data={
                        gesture.word: (gesture.left_hand, gesture.right_hand)
                    },
                )
                gesture_dataset.analyze_videos(
                    csv_out_path=gesture.videos_location / "dataset.csv"
                )
                gesture_dataset.load_from_csv(gesture.videos_location / "dataset.csv")
                pprint(Classifier().gesture_classifier.gesture_dataset.reverse_lookup_dict)
                if gesture.word in Classifier().gesture_classifier.gesture_dataset.reverse_lookup_dict:
                    print("Gesture already exists. It was probably updated.")
                    Classifier().gesture_classifier.update_gesture_dataset(gesture_dataset)
                else:
                    Classifier().gesture_classifier.append_dataset(gesture_dataset)

            Classifier().gesture_classifier.train(save_path=SAVED_MODEL_PATH)

            for gesture in new_gestures:
                gesture.status = Gesture.Status.COMPLETE
                gesture.save()
                # Classifier().gesture_classifier.gesture_dataset.add_django_gesture(gesture)
            IS_TRAINING = False
    except Exception as e:
        print(f"ERROR: Failed background training: {e}")
        IS_TRAINING = False
        raise e


def retrain_model():
    global IS_TRAINING
    print("Checking for new gestures...")
    if IS_TRAINING:
        print("Already retraining. Please wait.")
        return
    new_gestures = Gesture.objects.filter(Q(status=Gesture.Status.PENDING)).all()
    deleted_gestures = Gesture.objects.filter(Q(status=Gesture.Status.DELETED)).all()
    if new_gestures or deleted_gestures:
        threading.Thread(
            target=retrain_thread, daemon=False, args=(new_gestures, deleted_gestures)
        ).start()
    else:
        print("No changes to gestures.")


def run_continuously(self, interval=1):
    # https://stackoverflow.com/a/60244694/5165250
    cease_continuous_run = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                self.run_pending()
                time.sleep(interval)

    continuous_thread = ScheduleThread(daemon=True)
    continuous_thread.start()
    return cease_continuous_run


def start_scheduler():
    print("Starting the scheduler")
    schedule.Scheduler.run_continuously = run_continuously
    scheduler = schedule.Scheduler()
    scheduler.every(BACKGROUND_TRAINING_TIME).seconds.do(retrain_model)
    scheduler.run_continuously()
