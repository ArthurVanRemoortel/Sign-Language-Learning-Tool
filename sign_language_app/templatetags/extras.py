from django import template
from django.contrib.auth.models import User

from sign_language_app.utils import is_teacher_or_admin

register = template.Library()

#
# @register.filter
# def to_columns(iterable, cols):
#     result = [[] for _ in range(cols)]
#     diffs = len(iterable) % cols
#     print("diffs:", diffs)
#     prev_index = 0
#     i = 0
#     times_offset = 0
#     for value in iterable:
#         index = int(i / cols)
#         if index != prev_index:
#             if diffs > 0:
#                 index = prev_index
#                 diffs -= 1
#                 times_offset += 1
#         index -= times_offset
#         print(i, index)
#         result[index].append(value)
#         prev_index = index
#         i += 1
#
#     return result


@register.filter
def to_columns(iterable, cols):
    """
    source: https://stackoverflow.com/a/54802737/5165250
    Yield cols number of sequential chunks from iterable.
    """
    d, r = divmod(len(iterable), cols)
    for i in range(cols):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield iterable[si:si + (d + 1 if i < r else d)]


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
