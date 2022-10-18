from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)


def video_test(request):
    context = {}
    return render(request, "sign_language_app/video_test.html", context)
