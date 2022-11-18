from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)


def courses_overview(request):
    context = {}
    return render(request, "sign_language_app/courses/overview.html", context)


def exercise(request, exercise_id):
    context = {}
    return render(request, "sign_language_app/exercises/exercise.html", context)


@login_required
def video_test(request):
    context = {}
    return render(request, "sign_language_app/video_test.html", context)
