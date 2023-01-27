import sys
import threading

from django.apps import AppConfig

from learning_site.settings import BACKGROUND_TRAINING_TIME


class SignLanguageAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sign_language_app"

    def ready(self):
        if "runserver" not in sys.argv:
            return True
        import sign_language_app.background_tasks
        from sign_language_app.classifier import Classifier

        # TODO: Clean this up. Load the classifier in a background thread to make startup faster.
        def t():
            Classifier()

        threading.Thread(target=t).start()

        if BACKGROUND_TRAINING_TIME != -1:
            sign_language_app.background_tasks.start_scheduler()
