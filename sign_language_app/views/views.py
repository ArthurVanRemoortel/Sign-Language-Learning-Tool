from pprint import pprint

from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import render
from django.http import HttpResponse
from sign_language_app.models import *

def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)


def courses_overview(request):
    courses = Course.objects.all()
    user = request.user
    context = {
        "courses": courses,
        "completed_units": [attempt.unit for attempt in UnitAttempt.objects.filter(user=user)],
        "next_units": {course.id: course.get_next_unit(user) for course in courses}
    }
    pprint(context)
    return render(request, "sign_language_app/courses/overview.html", context)


def exercise(request, exercise_id):
    context = {}
    return render(request, "sign_language_app/exercises/exercise.html", context)


@login_required
def video_test(request):
    context = {}
    return render(request, "sign_language_app/video_test.html", context)
