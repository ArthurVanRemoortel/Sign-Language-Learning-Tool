import threading
import time
from pathlib import Path

from celery import shared_task
from django.db.models import Q
from django_celery_beat.models import PeriodicTask, IntervalSchedule

from sign_language_app.classifier import gesture_classifier
from sign_language_app.models import Gesture
from sl_ai.dataset import GestureDataset

IS_TRAINING = False


def retrain_thread(new_gestures):
    global IS_TRAINING
    IS_TRAINING = True
    print("Starting training...")
    print(new_gestures)
    csv_out_path = Path('sl_ai/uploaded_gestures_dataset.csv')
    model_path = Path('sl_ai/model.h5')

    new_dataset = GestureDataset()
    for gesture in new_gestures:
        gesture.status = Gesture.Status.TRAINING
        gesture.save()
        new_dataset.add_django_gesture(gesture)

    # TODO: Should return the data. Writing to csv should be be separate function.
    new_dataset.analyze_videos(csv_out_path=csv_out_path)

    for gesture in new_gestures:
        gesture.status = Gesture.Status.COMPLETE
        gesture.save()

    new_dataset.load_from_csv(csv_out_path)

    gesture_classifier.append_dataset(new_dataset)
    gesture_classifier.train(save_path=model_path)
    IS_TRAINING = False


def retrain_model():
    global IS_TRAINING
    new_gestures = Gesture.objects.filter(Q(status=Gesture.Status.PENDING) | Q(status=Gesture.Status.TRAINING)).all()
    if IS_TRAINING:
        print("Already retraining. Please wait.")
        return
    if new_gestures:
        threading.Thread(target=retrain_thread, daemon=False, args=(new_gestures, )).start()
    else:
        print("No new gestures.")



@shared_task(name="test")
def retrain_model_task(self):
    print(f"background task: test {self}")


def setup_training_task():
    schedule, created = IntervalSchedule.objects.get_or_create(
        every=1,
        period=IntervalSchedule.SECONDS,
    )

    try:
        task = PeriodicTask.objects.get(
            task="test",
        )
    except PeriodicTask.DoesNotExist:
        task = PeriodicTask(
            name="Model Training Task",
            task="test",
            interval=schedule,
        )
        task.save()
#
