import sys
from django.apps import AppConfig


class SignLanguageAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sign_language_app'

    def ready(self):
        if 'runserver' not in sys.argv:
            return True
        import sign_language_app.background_tasks
        print('Project is ready...')
        sign_language_app.background_tasks.setup_training_task()
