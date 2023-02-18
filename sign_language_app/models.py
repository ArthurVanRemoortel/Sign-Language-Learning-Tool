import os
import random
import shutil
from pathlib import Path
from typing import List, Optional

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q, QuerySet
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _

from learning_site.settings import (
    MEDIA_ROOT,
    USER_GESTURES_ROOT,
    UPLOADED_GESTURES_ROOT,
    VGT_GESTURES_ROOT,
)
from sl_ai.utils import clean_listdir


class UserSettings(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=False,
        related_name="settings",
    )
    allow_video_uploads = models.BooleanField(default=True)
    allow_sharing_with_teachers = models.BooleanField(default=True)
    allow_video_training = models.BooleanField(default=False)


class GestureLocation(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.name} (id={self.id})"


class Gesture(models.Model):
    class Status(models.TextChoices):
        PENDING = "0", _("Pending")
        TRAINING = "1", _("Training")
        COMPLETE = "2", _("Complete")
        DELETED = "3", _("Pending Deletion")

    word = models.CharField(max_length=100)
    locations = models.ManyToManyField(GestureLocation)
    left_hand = models.BooleanField(default=True)
    right_hand = models.BooleanField(default=True)
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        related_name="gestures",
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
        """Returns a string representation of gesture that includes the handedness of the gesture.
        Example: "Hello_01" where 0 indicates that the left hand is not used and 1 that the right hand is used.
        """
        return f"{self.word}_{1 if self.left_hand else 0}{1 if self.right_hand else 0}"

    @property
    def videos_location(self) -> Path:
        if self.creator:
            return Path(
                UPLOADED_GESTURES_ROOT / str(self.creator.id) / self.handed_string
            )
        else:
            return Path(VGT_GESTURES_ROOT / self.handed_string)

    @property
    def reference_video(self):
        """Returns the video path to a solution"""
        if self.creator:
            videos_location = (
                f"vgt-uploaded/{str(self.creator.id)}/{self.handed_string}"
            )
        else:
            videos_location = f"vgt-all/{self.handed_string}"
        video_file = random.choice(clean_listdir(MEDIA_ROOT / videos_location))
        return f"{videos_location}/{video_file}"

    def delete_videos(self):
        if not self.creator:
            return
        try:
            shutil.rmtree(
                UPLOADED_GESTURES_ROOT / str(self.creator.id) / self.handed_string
            )
        except OSError:
            pass


class Course(models.Model):
    class Difficulty(models.TextChoices):
        BEGINNER = "1", _("Beginner")
        INTERMEDIATE = "2", _("Intermediate")
        ADVANCED = "3", _("Advanced")

    class Visibility(models.TextChoices):
        PRIVATE = "1", _("Private")
        STUDENTS = "2", _("Students")
        PUBLIC = "3", _("Public")

    name = models.CharField(max_length=100)
    description = models.TextField(max_length=200)
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        related_name="courses",
    )

    class Meta:
        ordering = ["-id"]

    difficulty = models.CharField(
        max_length=2,
        choices=Difficulty.choices,
        default=Difficulty.BEGINNER,
    )

    visibility = models.CharField(
        max_length=2, choices=Visibility.choices, default=Visibility.PUBLIC
    )

    def get_next_unit(self, user: User):
        """Determines which unit the user is recommended to start next."""
        attempts = [
            attempt.unit.id for attempt in UnitAttempt.objects.filter(Q(user=user))
        ]
        for unit in self.units.all():
            if unit.id not in attempts:
                return unit

    @classmethod
    def get_accessible_by_user(cls, user: User):
        """Gets all courses are that are accessible by a provided user."""
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
    course = models.ForeignKey(
        Course, on_delete=models.CASCADE, related_name="units", null=False
    )
    ordering_number = models.IntegerField(null=False)

    class Meta:
        ordering = ["ordering_number"]

    @property
    def next_unit(self):
        """Determines which unit the user is recommended to start next."""
        course_units = self.course.units.all()
        for i, unit in enumerate(course_units):
            if unit.id == self.id:
                if i < len(course_units) - 1:
                    return course_units[i + 1]
        return None

    def __str__(self):
        return f"{self.name} ({self.course.name}) (id={self.id})"


class UnitAttempt(models.Model):
    """Represents a user attempting to complete a course. Used to keep a history and determine user progress."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=False,
        related_name="unit_attempts",
    )
    unit = models.ForeignKey(
        Unit, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    datetime = models.DateTimeField(null=False, auto_now_add=True)
    score = models.IntegerField(null=False, default=0)
    is_overruled = models.BooleanField(default=False, null=False)

    def __str__(self):
        return f"{self.unit} by {self.user}"

    # def get_user_gesture_videos(self) -> List[Path]:
    #     """Returns the videos files for this user and unit."""
    #     user_settings: UserSettings = self.user.settings.first()
    #     if not user_settings.allow_video_uploads:
    #         return list()
    #     videos_location: Path = USER_GESTURES_ROOT / self.unit.id / self.user.id
    #     return [
    #         videos_location / str(video_file)
    #         for video_file in clean_listdir(videos_location)
    #     ]

    @property
    def passed(self) -> bool:
        """Return if the user scored high enough to pass."""
        return self.score >= 50


class GestureAttempt(models.Model):
    """Represents a user attempting to perform a gesture. Used to calculate score and keep track of videos for teachers."""

    unit_attempt = models.ForeignKey(
        UnitAttempt,
        on_delete=models.CASCADE,
        null=False,
        related_name="gesture_attempts",
    )
    gesture = models.ForeignKey(Gesture, on_delete=models.CASCADE, null=False)
    attempt = models.IntegerField(
        null=False, default=0
    )  # How many attempts it took the user to complete this exercise. Determines the final score.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False
    )
    success = models.BooleanField(default=False, null=False)

    @property
    def attempts_video_url(self) -> Optional[str]:
        """Find the video of the user gesture in the temporary "last" folder."""
        user_settings: UserSettings = self.user.settings.first()
        if not user_settings.allow_video_uploads:
            return None
        location = (
            Path("vgt-users")
            / str(self.user.id)
            / str(self.unit_attempt.unit.id)
            / str(self.unit_attempt.id)
            / f"{self.gesture.id}_{self.attempt}.webm"
        )
        return location.as_posix() if (MEDIA_ROOT / location).exists() else None

    class Meta:
        ordering = ["id", "attempt"]


class TeacherCode(models.Model):
    """Represents the unique teacher invitation code."""

    # TODO: Move generate_teacher_code() here as a class method.
    code = models.CharField(max_length=100)
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=False,
        related_name="teacher_code",
    )


class StudentsAccess(models.Model):
    # TODO: Maybe rename this to Enrollment
    """Keeps track of students and teachers. Determines if students can access a teacher's courses."""
    student = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=False,
        related_name="teachers_access",
    )
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=False,
        related_name="students_access",
    )

    @classmethod
    def get_teachers(cls, student: User) -> QuerySet[User]:
        """Find all the teacher for a student. TODO: Makes more sense to put this an a custom user model."""
        if not student:
            return User.objects.none()
        teacher_ids = StudentsAccess.objects.filter(student=student).values_list(
            "teacher", flat=True
        )
        return User.objects.filter(Q(id__in=teacher_ids))

    @classmethod
    def get_students(cls, teacher: User) -> QuerySet[User]:
        """Find all the students for a teacher. TODO: Makes more sense to put this an a custom user model."""
        if not teacher:
            return User.objects.none()
        student_ids = StudentsAccess.objects.filter(teacher=teacher).values_list(
            "student", flat=True
        )
        return User.objects.filter(Q(id__in=student_ids))

    @classmethod
    def get_school_courses(cls, student: User) -> QuerySet[Course]:
        """Find all the courses for a student that are published by all their teachers. TODO: Makes more sense to put this an a custom user model."""
        if not student:
            return User.objects.none()
        teachers = StudentsAccess.get_teachers(student)
        teacher_courses = Course.objects.filter(Q(creator__in=teachers))
        return teacher_courses


@receiver(post_delete, sender=Gesture)
def signal_function_name(sender, instance: Gesture, using, **kwargs):
    """Delete videos on disk when a gesture id deleted."""
    instance.delete_videos()
