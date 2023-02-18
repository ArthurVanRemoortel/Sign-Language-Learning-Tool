import os
import random
import shutil
import string
from functools import wraps
from pathlib import Path
from typing import List, Tuple

from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.http import Http404
from rolepermissions.checkers import has_permission, has_role
from rolepermissions.decorators import has_role_decorator
from rolepermissions.utils import user_is_authenticated

from learning_site.roles import Teacher
from learning_site.settings import USER_GESTURES_ROOT
from sign_language_app.models import TeacherCode, Gesture, Unit, Course, UnitAttempt


def get_user(request):
    return request.user if not request.user.is_anonymous else None


def is_teacher(user: User):
    if not user:
        return False
    return has_role(user, Teacher)


def is_admin(user: User):
    if not user:
        return False
    return user.is_superuser


def is_teacher_or_admin(user: User) -> bool:
    """Tests if a user is an admin or a teacher."""
    if not user:
        return False
    return is_admin(user) or is_teacher(user)


def teacher_or_admin_required(function):
    """Decorator used to limit access to a view to admins or teachers."""

    @wraps(function)
    def wrap(request, *args, **kwargs):
        user = get_user(request)
        if is_teacher_or_admin(user=user):
            return function(request, *args, **kwargs)
        raise PermissionDenied()

    return wrap


def generate_teacher_code() -> str:
    """generates a unique teacher invitation code."""
    code = None
    while code is None or TeacherCode.objects.filter(code=code).exists():
        code = "".join(random.sample(string.ascii_uppercase, 5))
    return code


def save_user_gesture_video(
    video_data, user: User, unit: Unit, gesture: Gesture, attempt: int
):
    """Saved a video blob to a temporary location."""
    location = USER_GESTURES_ROOT / str(user.id) / str(unit.id) / "last"
    location.mkdir(parents=True, exist_ok=True)
    with open(location / f"{gesture.id}_{attempt}.webm", "wb+") as file_object:
        for chunk in video_data.chunks():
            file_object.write(chunk)


def save_lase_attempts_videos(user: User, unit: Unit, unit_attempt: UnitAttempt):
    temp_location = USER_GESTURES_ROOT / str(user.id) / str(unit.id) / 'last'
    new_location = USER_GESTURES_ROOT / str(user.id) / str(unit.id) / str(unit_attempt.id)
    shutil.copytree(temp_location, new_location)
    shutil.rmtree(temp_location)


def copy_user_gesture_video(user: User, unit: Unit, gesture: Gesture, attempt: int, unit_attempt: UnitAttempt):
    location = (
        USER_GESTURES_ROOT
        / str(user.id)
        / str(unit.id)
        / str(unit_attempt.id)
        / f"{gesture.id}_{attempt}.webm"
    )
    if location.exists():
        copy_location = USER_GESTURES_ROOT / str(user.id) / str(unit.id) / "saved"
        copy_location.mkdir(parents=True, exist_ok=True)
        copy_location = copy_location / f"{gesture.id}_{attempt}.webm"
        shutil.move(location, copy_location)


def find_course_recommendations(
    user, max_courses=4
) -> (str, List[Tuple[Course, Unit]]):
    if not user:
        # User is not logged in. Select some beginner courses.
        return "Courses for beginners", [
            (c, c.units.first())
            for c in Course.objects.filter(
                Q(difficulty=Course.Difficulty.BEGINNER)
            ).all()[:max_courses]
        ]

    recommendations = []
    for c in Course.objects.filter(Q(units__unit_attempts__user=user)).distinct()[
        :max_courses
    ]:
        next_unit = c.get_next_unit(user)
        if next_unit:
            recommendations.append((c, next_unit))
    return "Continue where you left of", recommendations
