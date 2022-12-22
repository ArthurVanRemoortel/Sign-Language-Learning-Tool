from functools import wraps

from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.http import Http404
from rolepermissions.checkers import has_permission, has_role
from rolepermissions.decorators import has_role_decorator
from rolepermissions.utils import user_is_authenticated

from learning_site.roles import Teacher


def get_user(request):
    return request.user if not request.user.is_anonymous else None


def is_teacher(user: User):
    return has_role(user, Teacher)


def is_admin(user: User):
    return user.is_superuser


def is_teacher_or_admin(user: User):
    return is_admin(user) or is_teacher(user)



def teacher_or_admin_required(function):
    @wraps(function)
    def wrap(request, *args, **kwargs):
        user = get_user(request)
        if is_teacher_or_admin(user=user):
            return function(request, *args, **kwargs)
        raise PermissionDenied()

    return wrap
