from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _


class GestureLocation(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.name} (id={self.id})"


class Gesture(models.Model):
    class Status(models.TextChoices):
        PENDING = '0', _('Pending')
        TRAINING = '1', _('Training')
        COMPLETE = '2', _('Complete')

    word = models.CharField(max_length=100)
    locations = models.ManyToManyField(GestureLocation)
    left_hand = models.BooleanField(default=True)
    right_hand = models.BooleanField(default=True)
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, related_name="gestures"
    )
    status = models.CharField(
        max_length=1,
        choices=Status.choices,
        default=Status.COMPLETE,
    )

    def __str__(self):
        return f"{self.word} (id={self.id})"

    @property
    def handed_string(self):
        return f"{self.word}_{1 if self.left_hand else 0}{1 if self.right_hand else 0}"


class Course(models.Model):
    class Difficulty(models.TextChoices):
        BEGINNER = '1', _('Beginner')
        INTERMEDIATE = '2', _('Intermediate')
        ADVANCED = '3', _('Advanced')

    class Visibility(models.TextChoices):
        PRIVATE = '1', _('Private')
        STUDENTS = '2', _('Students')
        PUBLIC = '3', _('Public')

    name = models.CharField(max_length=100)
    description = models.TextField(max_length=200)
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, related_name="courses"
    )

    class Meta:
        ordering = ["-id"]

    difficulty = models.CharField(
        max_length=2,
        choices=Difficulty.choices,
        default=Difficulty.BEGINNER,
    )

    visibility = models.CharField(
        max_length=2,
        choices=Visibility.choices,
        default=Visibility.PUBLIC
    )

    def get_difficulty(self) -> Difficulty:
        # Get value from choices enum
        return self.Difficulty[self.difficulty]

    def get_next_unit(self, user: User):
        attempts = [attempt.unit.id for attempt in UnitAttempt.objects.filter(Q(user=user))]
        for unit in self.units.all():
            if unit.id not in attempts:
                return unit

    def __str__(self):
        return f"{self.name} (id={self.id})"


class Unit(models.Model):
    name = models.CharField(max_length=100)
    gestures = models.ManyToManyField(Gesture)
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="units", null=False)
    ordering_number = models.IntegerField(null=False)

    class Meta:
        ordering = ["ordering_number"]

    def __str__(self):
        return f"{self.name} ({self.course.name}) (id={self.id})"


class UnitAttempt(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    unit = models.ForeignKey(
        Unit, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    datetime = models.DateTimeField(null=False)

    def __str__(self):
        return f"{self.unit} by {self.user}"
