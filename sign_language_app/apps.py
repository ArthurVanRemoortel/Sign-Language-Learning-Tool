import os
import sys
import threading

from django.apps import AppConfig

from learning_site.settings import BACKGROUND_TRAINING_TIME


class SignLanguageAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sign_language_app"

    def ready(self):
        server_version = os.environ.get("SERVER_SOFTWARE", "")

        if "gunicorn" not in server_version and "runserver" not in sys.argv:
            print(1)
            return

        if "runserver" in sys.argv and (
            os.environ.get("RUN_MAIN") != "true" and "--noreload" not in sys.argv
        ):
            print("runserver" in sys.argv)
            print(os.environ.get("RUN_MAIN") != "true")
            print("--noreload" not in os.environ)
            print(os.environ)
            print(2)
            return

        import sign_language_app.background_tasks
        from sign_language_app.classifier import Classifier

        print("Startup...")

        # TODO: Clean this up. Load the classifier in a background thread to make startup faster.
        def t():
            Classifier().load_dataset()

        threading.Thread(target=t, daemon=True).start()
        # t()

        if BACKGROUND_TRAINING_TIME != -1:
            sign_language_app.background_tasks.start_scheduler()
