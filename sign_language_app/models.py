from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _


class Gesture(models.Model):
    word = models.CharField(max_length=100)


class Course(models.Model):
    class Difficulty(models.TextChoices):
        BEGINNER = '1', _('Beginner')
        INTERMEDIATE = '2', _('Intermediate')
        ADVANCED = '3', _('Advanced')

    name = models.CharField(max_length=100)
    description = models.TextField(max_length=200)
    is_public = models.BooleanField(default=True)

    difficulty = models.CharField(
        max_length=2,
        choices=Difficulty.choices,
        default=Difficulty.BEGINNER,
    )

    def get_difficulty(self) -> Difficulty:
        # Get value from choices enum
        return self.Difficulty[self.difficulty]

    def get_next_unit(self, user: User):
        attempts = [attempt.unit.id for attempt in UnitAttempt.objects.filter(Q(user=user))]
        for unit in self.units.all():
            if unit.id not in attempts:
                return unit



class Unit(models.Model):
    name = models.CharField(max_length=100)
    gestures = models.ManyToManyField(Gesture)
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="units", null=False)
    ordering_number = models.IntegerField(null=False)

    class Meta:
        ordering = ["ordering_number"]


class UnitAttempt(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    unit = models.ForeignKey(
        Unit, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    datetime = models.DateTimeField(null=False)
