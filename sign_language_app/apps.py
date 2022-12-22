import sys
from django.apps import AppConfig


class SignLanguageAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sign_language_app'

    def ready(self):
        if 'runserver' not in sys.argv:
            return True
        import sign_language_app.background_tasks
        from sign_language_app.classifier import Classifier
        print('Project is ready...')
        # sign_language_app.background_tasks.setup_training_task()
        sign_language_app.background_tasks.start_scheduler()
        Classifier()
