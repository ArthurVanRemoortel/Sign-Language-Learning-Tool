from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)
