import os
import random
from pathlib import Path
from typing import List, Optional

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q, QuerySet
from django.utils.translation import gettext_lazy as _

from learning_site.settings import MEDIA_ROOT, USER_GESTURES_ROOT


class UserSettings(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="settings"
    )
    allow_video_uploads = models.BooleanField(default=False)



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

    @property
    def next_unit(self):
        """ Determines which unit the user is recommended to start next.  """
        course_units = self.course.units.all()
        for i, unit in enumerate(course_units):
            if unit.id == self.id:
                if i < len(course_units) - 1:
                    return course_units[i + 1]
        return None

    def __str__(self):
        return f"{self.name} ({self.course.name}) (id={self.id})"


class UnitAttempt(models.Model):
    """ Represents a user attempting to complete a course. Used to keep a history and determine user progress. """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    unit = models.ForeignKey(
        Unit, on_delete=models.CASCADE, null=False, related_name="unit_attempts"
    )
    datetime = models.DateTimeField(null=False)
    score = models.IntegerField(null=False, default=0)

    def __str__(self):
        return f"{self.unit} by {self.user}"

    def get_user_gesture_videos(self) -> List[Path]:
        """ Returns the videos files for this user and unit. """
        user_settings: UserSettings = self.user.settings
        if not user_settings.allow_video_uploads:
            return list()
        videos_location: Path = USER_GESTURES_ROOT / self.unit.id / self.user.id
        return [videos_location / str(video_file) for video_file in os.listdir(videos_location)]

    @property
    def calculate_score(self) -> int:
        """ Calculated the score for this unit based on the number if attempts for each gesture. """
        points = 0
        max_points = 0
        gesture_attempts = self.gesture_attempts.all()
        for gesture_attempt in gesture_attempts:
            points += gesture_attempt.attempts
            max_points += 3
        if max_points == 0:
            return 100
        return int(points / max_points * 100)


class GestureAttempt(models.Model):
    """ Represents a user attempting to perform a gesture. Used to calculate score and keep track of videos for teachers. """
    unit_attempt = models.ForeignKey(UnitAttempt, on_delete=models.CASCADE, null=False, related_name="gesture_attempts")
    gesture = models.ForeignKey(Gesture, on_delete=models.CASCADE, null=False)
    attempts = models.IntegerField(null=False, default=0)  # How many attempts it took the user to complete this exercise. Determines the final score.

    def get_video_path(self) -> Optional[Path]:
        user_settings: UserSettings = self.unit_attempt.user.settings
        if not user_settings.allow_video_uploads:
            return None


class TeacherCode(models.Model):
    """ Represents the unique teacher invitation code. """
    # TODO: Move generate_teacher_code() here as a class method.
    code = models.CharField(max_length=100)
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="teacher_code"
    )


class StudentsAccess(models.Model):
    """ Keeps track of students and teachers. Determines if students can access a teacher's courses. """
    student = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="teachers_access"
    )
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=False, related_name="students_access"
    )

    @classmethod
    def get_teachers(cls, student: User) -> QuerySet[User]:
        """ Find all the teacher for a student. TODO: Makes more sense to put this an a custom user model. """
        if not student:
            return User.objects.none()
        teacher_ids = StudentsAccess.objects.filter(student=student).values_list('teacher', flat=True)
        return User.objects.filter(Q(id__in=teacher_ids))

    @classmethod
    def get_students(cls, teacher: User) -> QuerySet[User]:
        """ Find all the students for a teacher. TODO: Makes more sense to put this an a custom user model. """
        if not teacher:
            return User.objects.none()
        student_ids = StudentsAccess.objects.filter(teacher=teacher).values_list('student', flat=True)
        return User.objects.filter(Q(id__in=student_ids))

    @classmethod
    def get_school_courses(cls, student: User) -> QuerySet[Course]:
        """ Find all the courses for a student that are published by all their teachers. TODO: Makes more sense to put this an a custom user model. """
        if not student:
            return User.objects.none()
        teachers = StudentsAccess.get_teachers(student)
        teacher_courses = Course.objects.filter(Q(creator__in=teachers))
        return teacher_courses
