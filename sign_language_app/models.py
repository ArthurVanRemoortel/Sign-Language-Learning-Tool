from django.db import models
from django.utils.translation import gettext_lazy as _


class Gesture(models.Model):
    word = models.CharField(max_length=100)


class Course(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(max_length=200)


class Unit(models.Model):
    class Difficulty(models.TextChoices):
        BEGINNER = '1', _('Beginner')
        INTERMEDIATE = '2', _('Intermediate')
        ADVANCED = '3', _('Advanced')

    models.CharField(max_length=100)
    gestures = models.ManyToManyField(Gesture)

    difficulty = models.CharField(
        max_length=2,
        choices=Difficulty.choices,
        default=Difficulty.BEGINNER,
    )

    def get_difficulty(self) -> Difficulty:
        # Get value from choices enum
        return self.Difficulty[self.difficulty]

