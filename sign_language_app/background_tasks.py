import os
import random
import threading
import time
from pathlib import Path

import schedule
from django.db.models import Q

from learning_site.settings import BACKGROUND_TRAINING_TIME
from sign_language_app.classifier import Classifier
from sign_language_app.models import Gesture
from sl_ai.dataset import GestureDataset

IS_TRAINING = False


def retrain_thread(new_gestures):
    global IS_TRAINING
    IS_TRAINING = True
    print("Starting training...")
    csv_out_path = Path('sl_ai/uploaded_gestures_dataset.csv')
    model_path = Path('sl_ai/model.h5')

    new_dataset = GestureDataset()

    # TODO: Should return the data. Writing to csv should be be separate function.
    new_dataset.analyze_videos(csv_out_path=csv_out_path)
    new_dataset.load_from_csv(csv_out_path)

    for gesture in new_gestures:
        gesture.status = Gesture.Status.COMPLETE
        gesture.save()

    # TODO: Add methods to Classifier()
    Classifier().gesture_classifier.append_dataset(new_dataset)
    Classifier().gesture_classifier.train(save_path=model_path)

    for gesture in new_gestures:
        gesture.status = Gesture.Status.TRAINING
        gesture.save()
        new_dataset.add_django_gesture(gesture)

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
