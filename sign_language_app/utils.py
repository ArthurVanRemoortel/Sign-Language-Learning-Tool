import random
import string
from functools import wraps
from pathlib import Path

from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.http import Http404
from rolepermissions.checkers import has_permission, has_role
from rolepermissions.decorators import has_role_decorator
from rolepermissions.utils import user_is_authenticated

from learning_site.roles import Teacher
from learning_site.settings import USER_GESTURES_ROOT
from sign_language_app.models import TeacherCode, Gesture, Unit


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
    """ Tests if a user is an admin or a teacher. """
    if not user:
        return False
    return is_admin(user) or is_teacher(user)


def teacher_or_admin_required(function):
    """ Decorator used to limit access to a view to admins or teachers."""
    @wraps(function)
    def wrap(request, *args, **kwargs):
        user = get_user(request)
        if is_teacher_or_admin(user=user):
            return function(request, *args, **kwargs)
        raise PermissionDenied()
    return wrap


def generate_teacher_code() -> str:
    """ generates a unique teacher invitation code. """
    code = None
    while code is None or TeacherCode.objects.filter(code=code).exists():
        code = ''.join(random.sample(string.ascii_uppercase, 5))
    return code


def save_user_gesture_video(video_data, user: User, unit: Unit, gesture: Gesture, attempt: int):
    """ Saved a video blob to a temporary location. """
    location = USER_GESTURES_ROOT / str(user.id) / "last" / str(unit.id)
    location.mkdir(parents=True, exist_ok=True)
    with open(location / f'{gesture.id}_{attempt}.webm', "wb+") as file_object:
        for chunk in video_data.chunks():
            file_object.write(chunk)
