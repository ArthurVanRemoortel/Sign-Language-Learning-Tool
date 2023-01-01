import os
import random
from typing import List

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q, QuerySet
from django.utils.translation import gettext_lazy as _

from learning_site.settings import MEDIA_ROOT


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
        """ Returns a string representation of gesture that includes the handedness of the gesture.
         Example: "Hello_01" where 0 indicates that the left hand is not used and 1 that the right hand is used.
         """
        return f"{self.word}_{1 if self.left_hand else 0}{1 if self.right_hand else 0}"

    @property
    def reference_video(self):
        if self.creator:
            videos_location = f'vgt-uploaded/{str(self.creator.id)}/{self.handed_string}'
        else:
            videos_location = f'vgt-all/{self.handed_string}'

        video_file = random.choice(os.listdir(MEDIA_ROOT / videos_location))
        return f'{videos_location}/{video_file}'


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

    def get_next_unit(self, user: User):
        """ Determines which unit the user is recommended to start next.  """
        attempts = [attempt.unit.id for attempt in UnitAttempt.objects.filter(Q(user=user))]
        for unit in self.units.all():
            if unit.id not in attempts:
                return unit

    @classmethod
    def get_accessible_by_user(cls, user: User):
        """ Gets all courses are that are accessible by a provided user. """
        public_courses = Course.objects.filter(visibility=Course.Visibility.PUBLIC)
        if not user:
            return public_courses
        user_courses = Course.objects.filter(creator=user)
        teacher_courses = StudentsAccess.get_school_courses(student=user)
        return user_courses | teacher_courses | public_courses

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


class TeacherCode(models.Model):
    code = models.CharField(max_length=100)

    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="teacher_code"
    )


class StudentsAccess(models.Model):
    student = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="teachers_access"
    )
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="students_access"
    )

    @classmethod
    def get_teachers(cls, student: User) -> QuerySet[User]:
        if not student:
            return User.objects.none()
        teacher_ids = StudentsAccess.objects.filter(student=student).values_list('teacher', flat=True)
        return User.objects.filter(Q(id__in=teacher_ids))

    @classmethod
    def get_students(cls, teacher: User) -> QuerySet[User]:
        if not teacher:
            return User.objects.none()
        student_ids = StudentsAccess.objects.filter(teacher=teacher).values_list('student', flat=True)
        return User.objects.filter(Q(id__in=student_ids))

    @classmethod
    def get_school_courses(cls, student: User) -> QuerySet[Course]:
        if not student:
            return User.objects.none()
        teachers = StudentsAccess.get_teachers(student)
        teacher_courses = Course.objects.filter(Q(creator__in=teachers))
        return teacher_courses
