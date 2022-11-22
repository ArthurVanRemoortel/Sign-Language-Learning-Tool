from pprint import pprint
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import render
from django.http import HttpResponse
from django.core.paginator import Paginator

from sign_language_app.forms import CoursesForm
from sign_language_app.models import *


def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)


def courses_overview(request, page=1):
    search_form = CoursesForm(request.POST)
    courses = Course.objects.all()
    if request.method == "GET":
        ...
    else:
        if search_form.is_valid():
            search_input_text = search_form.cleaned_data.get("search_input", None)
            if search_input_text:
                courses = courses.filter(Q(name__icontains=search_input_text) | Q(description__icontains=search_input_text) | Q(units__name__icontains=search_input_text)).distinct()
    user = request.user

    courses = courses.union(*[courses for _ in range(100)], all=True)  # Hacky way to make the results longer to test pagination.

    paginator = Paginator(courses, 15)
    page_number = request.GET.get('page', 1)
    courses_paginator = paginator.get_elided_page_range(page_number, on_each_side=2, on_ends=1)
    courses = paginator.get_page(page_number)

    context = {
        "courses": courses,
        "search_form": search_form,
        'courses_paginator': courses_paginator,
        "completed_units": [attempt.unit for attempt in UnitAttempt.objects.filter(user=user)],
        "next_units": {course.id: course.get_next_unit(user) for course in courses}
    }
    return render(request, "sign_language_app/courses_overview.html", context)


def unit_view(request, unit_id):
    unit = get_object_or_404(Unit, pk=unit_id)
    context = {
        'unit': unit
    }
    return render(request, "sign_language_app/unit.html", context)


@login_required
def video_test(request):
    context = {}
    return render(request, "sign_language_app/video_test.html", context)
