from __future__ import absolute_import, unicode_literals

from celery import Celery

import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'learning_site.settings')

app = Celery('learning_site')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.conf.broker_url = 'redis://localhost:6379/0'

app.autodiscover_tasks()

app.conf.beat_schedule = {

}


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')