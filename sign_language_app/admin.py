from django.contrib import admin
from sign_language_app.models import *
# Register your models here.

admin.site.register(Course)
admin.site.register(Unit)
admin.site.register(Gesture)
admin.site.register(UnitAttempt)
