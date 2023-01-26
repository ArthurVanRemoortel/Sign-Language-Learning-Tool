import os
import random
import threading
import time
from pathlib import Path
from typing import List

import schedule
from django.db.models import Q

from learning_site.settings import BACKGROUND_TRAINING_TIME, UPLOADED_GESTURES_ROOT
from sign_language_app.classifier import Classifier
from sign_language_app.models import Gesture
from sl_ai.dataset import GestureDataset

IS_TRAINING = False


def retrain_thread(new_gestures: List[Gesture]):
    if not new_gestures:
        return
    global IS_TRAINING
    IS_TRAINING = True
    print("Starting training...")
    csv_out_path = Path('sl_ai/uploaded_gestures_dataset.csv')
    model_path = Path('sl_ai/model.h5')
    # new_dataset = GestureDataset()
    # new_dataset.scan_videos(UPLOADED_GESTURES_ROOT, handedness_data={g.word: (g.left_hand, g.right_hand) for g in new_gestures})
    # new_dataset.analyze_videos(csv_out_path=csv_out_path)
    # if not csv_out_path.exists():
        # Nothing was created.
        # IS_TRAINING = False
        # return
    # new_dataset.load_from_csv(csv_out_path)

    for gesture in new_gestures:
        if not gesture.creator:
            print("Gesture has no creator. This should not happen.")
        gesture.status = Gesture.Status.TRAINING
        gesture.save()
    for gesture in new_gestures:
        if not gesture.creator:
            print("Gesture has no creator. This should not happen.")
        gesture_dataset = GestureDataset(single_gesture=True)
        gesture_dataset.scan_videos(gesture.videos_location, handedness_data={gesture.word: (gesture.left_hand, gesture.right_hand)})
        gesture_dataset.analyze_videos(csv_out_path=gesture.videos_location / "dataset.csv")
        gesture_dataset.load_from_csv(gesture.videos_location / "dataset.csv")
        Classifier().gesture_classifier.append_dataset(gesture_dataset)

    Classifier().gesture_classifier.train(save_path=model_path)

    for gesture in new_gestures:
        gesture.status = Gesture.Status.COMPLETE
        gesture.save()
        Classifier().gesture_classifier.gesture_dataset.add_django_gesture(gesture)

    IS_TRAINING = False


def retrain_model():
    global IS_TRAINING
    print("Checking for new gestures...")
    if IS_TRAINING:
        print("Already retraining. Please wait.")
        return
    new_gestures = Gesture.objects.filter(Q(status=Gesture.Status.PENDING)).all()
    if new_gestures:
        threading.Thread(target=retrain_thread, daemon=False, args=(new_gestures, )).start()
    else:
        print("No new gestures.")


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
