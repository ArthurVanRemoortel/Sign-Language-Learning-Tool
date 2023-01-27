from django import template
from django.contrib.auth.models import User

from sign_language_app.utils import is_teacher_or_admin

register = template.Library()


@register.filter
def to_columns(iterable, cols):
    """
    source: https://stackoverflow.com/a/54802737/5165250
    Yield cols number of sequential chunks from iterable.
    """
    d, r = divmod(len(iterable), cols)
    for i in range(cols):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield iterable[si : si + (d + 1 if i < r else d)]


@register.filter
def index(indexable, i):
    return indexable[i]


@register.filter
def sort(iterable, attr):
    return sorted(iterable, key=lambda item: getattr(item, attr))


@register.filter
def get_value(dictionary, key):
    return dictionary.get(key, None)


@register.filter
def check_teacher_or_admin(user: User):
    return is_teacher_or_admin(user)


@register.filter
def times(number):
    return range(number)
