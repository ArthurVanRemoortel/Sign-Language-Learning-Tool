from __future__ import absolute_import, unicode_literals

from celery import Celery
from datetime import datetime, timedelta

import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'learning_site')

app = Celery('learning_site')

app.config_from_object('django.conf:settings', namespace='CELERY')
