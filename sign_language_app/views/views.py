import json
from pprint import pprint

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Q, F
from django.shortcuts import render
from django.http import HttpResponse, Http404, JsonResponse, HttpResponseNotFound
from django.core.paginator import Paginator, EmptyPage
from django.urls import reverse
from django.views.defaults import page_not_found

from sign_language_app import serializers, utils
from sign_language_app.forms import CoursesForm
from sign_language_app.models import *
from django.core import serializers as django_serializers

from sign_language_app.utils import get_user, find_course_recommendations, copy_user_gesture_video, \
    save_lase_attempts_videos, is_admin
from sign_language_app.views.error_views import error_view


def index(request):
    context = {}
    return render(request, "sign_language_app/index.html", context)


def courses_overview(request):
    search_form = CoursesForm(request.GET)
    user = get_user(request)
    courses = Course.get_accessible_by_user(user=user)
    page_number = request.GET.get("page", 1)
    filters = request.GET.get("filters", None)
    if search_form.is_valid():
        search_input_text = search_form.cleaned_data.get("search_input", None)
        if search_input_text:
            courses = courses.filter(
                Q(name__icontains=search_input_text)
                | Q(description__icontains=search_input_text)
                | Q(units__name__icontains=search_input_text)
            ).distinct()

    if filters:
        for filter_query in filters.split(","):
            if "difficulty_" in filter_query:
                courses = courses.filter(
                    Q(difficulty=Course.Difficulty[filter_query.split("_")[-1].upper()])
                )
            elif filter_query == "continue":
                if not user:
                    courses = Course.objects.none()
                else:
                    courses = courses.filter(
                        Q(units__unit_attempts__user=user)
                    ).distinct()
            elif filter_query == "saved":
                # TODO: Not implemented yet.
                if not user:
                    courses = Course.objects.none()
                else:
                    courses = Course.objects.none()
            elif filter_query == "school":
                if not user:
                    courses = Course.objects.none()
                else:
                    courses = StudentsAccess.get_school_courses(student=user)

    completed_units = []
    next_units = {}
    if user:
        completed_units = [
            attempt.unit for attempt in UnitAttempt.objects.filter(user=user)
        ]
        next_units = {course.id: course.get_next_unit(user) for course in courses}

    recommendation_string, recommended_courses = find_course_recommendations(
        user, max_courses=4
    )

    paginator = Paginator(courses, 15)
    try:
        courses = paginator.page(page_number)
    except EmptyPage as e:
        return error_view(request, exception="That page number is to high.")
    courses_paginator = paginator.get_elided_page_range(
        page_number, on_each_side=2, on_ends=1
    )
    context = {
        "courses": courses,
        "search_form": search_form,
        "page_number": page_number,
        "filters": filters,
        "courses_paginator": courses_paginator,
        "completed_units": completed_units,
        "next_units": next_units,
        "recommended_courses": recommended_courses,
        "recommendation_string": recommendation_string,
    }
    return render(request, "sign_language_app/courses_overview.html", context)


@login_required
def unit_view(request, unit_id):
    user = get_user(request)
    unit = get_object_or_404(Unit, pk=unit_id)
    context = {
        "unit": unit,
        "user_settings": user.settings.first(),
        "gestures": json.dumps(
            serializers.GestureSerializer(unit.gestures.all(), many=True).data
        ),
        "show_landmarks": is_admin(user)
    }
    return render(request, "sign_language_app/unit.html", context)


@login_required
def unit_summary(request, unit_id, attempt_id):
    user = get_user(request)
    unit = get_object_or_404(Unit, pk=unit_id)
    this_attempt = get_object_or_404(UnitAttempt, pk=attempt_id)
    unit_attempts = UnitAttempt.objects.filter(user=user, unit=unit).all()
    if this_attempt.unit != unit:
        return error_view(request, exception="This unit attempts does not exist.")
    if not unit_attempts:
        return error_view(request, exception="You have never completed this course.")
    if this_attempt.user != user:
        return error_view(request, exception="You do not have access this user's data")
    context = {
        "unit": unit,
        "this_attempt": this_attempt,
        "user_settings": user.settings.first(),
        "attempts_per_gesture": {
            gesture: [
                gesture_attempt
                for gesture_attempt in this_attempt.gesture_attempts.filter(
                    gesture=gesture
                )
            ]
            for gesture in unit.gestures.all()
        },
        "unit_attempts": unit_attempts
    }
    return render(request, "sign_language_app/unit_summary.html", context)


@login_required
def save_unit_attempt(request, unit_id: int):
    user = get_user(request)
    user_settings: UserSettings = user.settings.first()
    unit: Unit = get_object_or_404(Unit, pk=unit_id)
    unit_attempt: UnitAttempt = UnitAttempt(user=user, unit=unit, score=0)
    unit_attempt.save()
    points = 0
    if request.method == "GET":
        messages.error(
            request, message="Something went wrong when saving your progress."
        )
        return redirect("index")

    data_from_post = json.load(request)
    for gesture_id, attempts in data_from_post["gestures"].items():
        for attempts_i, attempt_success in enumerate(attempts, start=1):
            gesture_attempt = GestureAttempt(
                unit_attempt=unit_attempt,
                gesture=Gesture.objects.get(pk=gesture_id),
                attempt=attempts_i,
                success=attempt_success,
                user=user,
            )
            gesture_attempt.save()
        has_correct_answer = any(filter(lambda success: success, attempts))
        if has_correct_answer and len(attempts) == 1:
            # Success without hints.
            points += 1
        elif has_correct_answer and len(attempts) == 2:
            # Success without hints.
            points += 0.75
        elif has_correct_answer and len(attempts) == 3:
            # Success with hint.
            points += 0.5
        else:
            # Never success.
            points += 0
    unit_attempt.score = int(points / unit.gestures.count() * 100)
    unit_attempt.save()
    print(f"Finished with a score of {unit_attempt.score}")
    if user_settings.allow_video_uploads:
        print("Copying videos:")
        save_lase_attempts_videos(user=user, unit=unit, unit_attempt=unit_attempt)
    return JsonResponse(
        {
            "continue": True,
            "redirect_url": reverse(
                "unit_summary",
                kwargs={"unit_id": unit.id, "attempt_id": unit_attempt.id},
            ),
        }
    )


@login_required
def upload_gesture_video(request):
    user = get_user(request)
    if request.method == "POST":
        video_data = request.FILES["video"]
        unit_id = request.POST["unit_id"]
        gesture_id = request.POST["gesture_id"]
        attempt = request.POST["attempt"]
        unit = get_object_or_404(Unit, pk=unit_id)
        gesture = get_object_or_404(Gesture, pk=gesture_id)
        # TODO: Check user settings.
        utils.save_user_gesture_video(
            video_data=video_data,
            unit=unit,
            gesture=gesture,
            user=user,
            attempt=attempt,
        )

    return JsonResponse({"status": "OK"})
