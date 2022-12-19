from pprint import pprint
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db.models import Q, F
from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.core.paginator import Paginator, EmptyPage
from django.views.defaults import page_not_found

from sign_language_app.forms import CoursesForm
from sign_language_app.models import *
from django.core import serializers



def get_user(request):
    return request.user if not request.user.is_anonymous else None


def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)


def error_view(request, exception):
    if isinstance(exception, Exception):
        error_message = exception
    else:
        error_message = exception

    context = {
        'request': request,
        'error_message': error_message
    }
    return render(request, "sign_language_app/errors/404.html", context)


class JoinedField:
    pass


def courses_overview(request):
    search_form = CoursesForm(request.GET)
    courses = Course.objects.all()
    page_number = request.GET.get('page', 1)
    filters = request.GET.get('filters', None)

    if search_form.is_valid():
        search_input_text = search_form.cleaned_data.get("search_input", None)
        if search_input_text:
            courses = courses.filter(Q(name__icontains=search_input_text) | Q(description__icontains=search_input_text) | Q(units__name__icontains=search_input_text)).distinct()


    completed_units = []
    next_units = {}
    user = get_user(request)

    if filters:
        for filter_query in filters.split(','):
            if 'difficulty_' in filter_query:
                courses = courses.filter(Q(difficulty=Course.Difficulty[filter_query.split('_')[-1].upper()]))
            elif filter_query == "continue":
                if not user:
                    courses = Course.objects.none()
                else:
                    courses = courses.filter(Q(units__unit_attempts__user=user)).distinct()
            elif filter_query == "saved":
                # TODO: Not implemented yet.
                if not user:
                    courses = Course.objects.none()
                else:
                    courses = Course.objects.none()
            elif filter_query == "shared_with_me":
                # TODO: Not implemented yet.
                if not user:
                    courses = Course.objects.none()
                else:
                    courses = Course.objects.none()

    if user:
        completed_units = [attempt.unit for attempt in UnitAttempt.objects.filter(user=user)]
        next_units = {course.id: course.get_next_unit(user) for course in courses}


    courses = courses.union(*[courses for _ in range(50)], all=True)  # Hacky way to make the results longer to test pagination.

    paginator = Paginator(courses, 15)
    try:
        courses = paginator.page(page_number)
    except EmptyPage as e:
        return error_view(request, exception='That page number is to high.')

    courses_paginator = paginator.get_elided_page_range(page_number, on_each_side=2, on_ends=1)

    context = {
        "courses": courses,
        "search_form": search_form,
        "page_number": page_number,
        "filters": filters,
        'courses_paginator': courses_paginator,
        "completed_units": completed_units,
        "next_units": next_units
    }
    return render(request, "sign_language_app/courses_overview.html", context)


@login_required
def unit_view(request, unit_id):
    unit = get_object_or_404(Unit, pk=unit_id)
    context = {
        'unit': unit,
        'gestures': serializers.serialize("json", unit.gestures.all())
    }
    return render(request, "sign_language_app/unit.html", context)

