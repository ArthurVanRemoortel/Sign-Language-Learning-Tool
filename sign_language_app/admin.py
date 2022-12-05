from django.contrib import admin
from sign_language_app import models
# Register your models here.

admin.site.register(models.Course)
admin.site.register(models.Unit)
admin.site.register(models.Gesture)
admin.site.register(models.GestureLocation)
admin.site.register(models.UnitAttempt)
