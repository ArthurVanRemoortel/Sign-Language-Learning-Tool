import os
import shutil
from pprint import pprint

import ffmpeg
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.db.models import Q
from django.http import JsonResponse, HttpResponseForbidden, HttpResponseServerError
from django.core import exceptions
from django.shortcuts import render, redirect, get_object_or_404
import json

from rolepermissions.checkers import has_role
from rolepermissions.roles import get_user_roles

from learning_site import settings
from learning_site.roles import Teacher
from learning_site.settings import MEDIA_ROOT, USER_GESTURES_ROOT, UPLOADED_GESTURES_ROOT
from sign_language_app import forms
from sign_language_app.forms import (
    UploadGestureForm,
    NewCourseForm,
    TeacherCodeForm,
    UserSettingsForm,
    UserProfileForm,
)
from sign_language_app.models import (
    Gesture,
    Course,
    Unit,
    TeacherCode,
    StudentsAccess,
    UnitAttempt,
    GestureAttempt,
    UserSettings,
)
from sign_language_app.utils import (
    teacher_or_admin_required,
    is_teacher_or_admin,
    get_user,
    generate_teacher_code,
)
from sl_ai.utils import clean_listdir


@login_required
def profile_overview(request):
    user: User = get_user(request)
    user_settings: UserSettings = user.settings.first()
    if request.method == "POST":
        form = UserProfileForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data.get("username", None)
            avatar_file = form.cleaned_data.get('avatar', None)
            if User.objects.filter(Q(username=username) & ~Q(pk=user.id)).exists():
                messages.error(
                    request,
                    f"Username {username} is already in use.",
                )
                return redirect("profile")
            user = User.objects.get(pk=user.id)
            user.username = username
            if avatar_file:
                print(avatar_file, avatar_file.name)
                root_path = USER_GESTURES_ROOT / 'avatars'
                os.makedirs(root_path, exist_ok=True)
                with open(root_path / avatar_file.name, "wb+") as destination:
                    for chunk in avatar_file.chunks():
                        destination.write(chunk)
                user_settings.avatar = avatar_file.name
                user_settings.save()
            user.save()
    else:
        form = UserProfileForm()
        form.initial['username'] = user.username
    context = {"current_section": "overview", "form": form, "user_settings": user_settings}
    return render(request, "sign_language_app/profile/profile_overview.html", context)


@login_required
def profile_settings(request):
    user = get_user(request)
    user_settings: UserSettings = user.settings.first()
    form = UserSettingsForm(
        request.POST or None, request.FILES or None, instance=user_settings
    )
    if form.is_valid():
        form.save()
        if not user_settings.allow_video_uploads:
            ...
    context = {"current_section": "settings", "form": form}
    return render(request, "sign_language_app/profile/profile_settings.html", context)


@login_required
def delete_data(request, delete_account):
    user: User = get_user(request)
    user = User.objects.get(id=user.id)
    user_settings: UserSettings = user.settings.first()
    for user_directory in clean_listdir(UPLOADED_GESTURES_ROOT):
        if user_directory == str(user.id):
            os.remove(UPLOADED_GESTURES_ROOT / user_directory)
    for gesture in Gesture.objects.filter(Q(creator=user)):
        gesture.status = Gesture.Status.DELETED
        gesture.save()
        # The gesture will be deleted later.
    if delete_account:
        user_settings.delete()
        # All related object are deleted using SQL Cascade.
        logout(request)
        user.delete()
        messages.success(request, message="Completely deleted you account and associated data.")
        return redirect('index')
    else:
        user_settings.save()
        GestureAttempt.objects.filter(Q(user=user)).all().delete()
        UnitAttempt.objects.filter(Q(user=user)).all().delete()
        Course.objects.filter(Q(creator=user)).all().delete()
        StudentsAccess.objects.filter(Q(student=user) | Q(teacher=user)).all().delete()
        messages.success(request, message="Deleted all data associated with you.")
        return redirect('profile')


@login_required
@teacher_or_admin_required
def regenerate_teacher_code(request):
    """Regenerated the teacher's teacher code. The old code will no longer be valid but existing student will keep access."""
    teacher = get_user(request)
    TeacherCode.objects.filter(teacher=teacher).update(code=generate_teacher_code())
    return redirect("manage_students")


@login_required
@teacher_or_admin_required
def remove_student_from_classroom(request, student_id: int):
    """Teacher removes one of its students"""
    teacher = get_user(request)
    student = get_object_or_404(User, pk=student_id)
    StudentsAccess.objects.filter(student=student, teacher=teacher).delete()
    return redirect("manage_students")


@login_required
def remove_teacher(request, teacher_id: int):
    """Student removes a teacher."""
    student = get_user(request)
    teacher = get_object_or_404(User, pk=teacher_id)
    StudentsAccess.objects.filter(student=student, teacher=teacher).delete()
    return redirect("manage_teachers")


@login_required
@teacher_or_admin_required
def manage_students_view(request):
    teacher = get_user(request)
    if not teacher.teacher_code.exists():
        teacher_code = TeacherCode(code=generate_teacher_code(), teacher=teacher)
        teacher_code.save()

    context = {
        "current_section": "manage_students",
        "teacher": teacher,
        "students": StudentsAccess.get_students(teacher=teacher),
    }
    return render(
        request, "sign_language_app/profile/classroom/manage_students.html", context
    )


@login_required
@teacher_or_admin_required
def student_details_view(request, student_id: int):
    teacher = get_user(request)
    student = get_object_or_404(User, pk=student_id)
    access = StudentsAccess.objects.filter(Q(student=student, teacher=teacher))
    if not access.exists():
        messages.error(
            request,
            "You cannot access this students information.",
        )
        return redirect("manage_students")
    context = {
        "student_settings": student.settings.first(),
        "current_section": "manage_students",
        "teacher": teacher,
        "student": student,
    }
    return render(
        request, "sign_language_app/profile/classroom/student_details.html", context
    )


@login_required
def history_view(request):
    user = get_user(request)
    courses = Course.objects.filter(Q(units__unit_attempts__user=user)).distinct()
    units = Unit.objects.filter(Q(course__in=courses))
    context = {
        "courses": courses,
        "last_unit_attempts": {
            unit.id: unit.unit_attempts.filter(Q(user=user)).order_by("datetime").last()
            for unit in units
        },
        "user_settings": user.settings.first(),
        "current_section": "history",
        "user": user,
    }
    return render(request, "sign_language_app/profile/user_history.html", context)


@login_required
@teacher_or_admin_required
def student_details_unit_attempts_view(request, student_id: int, unit_attempts_id: int):
    teacher = get_user(request)
    student = get_object_or_404(User, pk=student_id)
    access = StudentsAccess.objects.filter(Q(student=student, teacher=teacher))
    unit_attempt = get_object_or_404(UnitAttempt, pk=unit_attempts_id)
    if not access.exists():
        messages.error(
            request,
            "You cannot access this students information.",
        )
        return redirect("manage_students")
    context = {
        "student_settings": student.settings.first(),
        "current_section": "manage_students",
        "teacher": teacher,
        "student": student,
        "unit_attempt": unit_attempt,
        "attempts_per_gesture": {
            gesture: [
                gesture_attempt
                for gesture_attempt in unit_attempt.gesture_attempts.filter(
                    gesture=gesture
                )
            ]
            for gesture in unit_attempt.unit.gestures.all()
        },
    }
    return render(
        request,
        "sign_language_app/profile/classroom/student_details_unit_attempts.html",
        context,
    )


@login_required
@teacher_or_admin_required
def overrule_gesture_attempt_view(
    request, student_id: int, unit_attempts_id: int, gesture_attempts_id: int
):
    if request.method == "GET":
        return HttpResponseForbidden()
    gesture_attempt = get_object_or_404(GestureAttempt, pk=gesture_attempts_id)
    unit_attempt = get_object_or_404(UnitAttempt, pk=unit_attempts_id)

    data_from_post = json.load(request)
    gesture_attempt.success = data_from_post["correct"]
    gesture_attempt.save()

    points = 0
    for gesture in unit_attempt.unit.gestures.all():
        gesture_attempts = unit_attempt.gesture_attempts.filter(Q(gesture=gesture))
        answers = list(map(lambda attempt: attempt.success, gesture_attempts))
        if len(answers) >= 1 and answers[0]:
            points += 1
        elif len(answers) >= 2 and answers[1]:
            points += 0.75
        elif len(answers) >= 3 and answers[2]:
            # Success with hint.
            points += 0.5
        else:
            # Never success.
            points += 0
    unit_attempt.score = int(points / unit_attempt.unit.gestures.count() * 100)
    unit_attempt.is_overruled = True
    unit_attempt.save()
    return JsonResponse({"status": "ok", "newScore": unit_attempt.score})


@login_required
@teacher_or_admin_required
def add_video_to_dataset(
    request, student_id: int, unit_attempts_id: int, gesture_attempts_id: int
):
    if request.method == "GET":
        return HttpResponseForbidden()

    gesture_attempt = get_object_or_404(GestureAttempt, pk=gesture_attempts_id)
    unit_attempt = get_object_or_404(UnitAttempt, pk=unit_attempts_id)
    teacher = get_user(request)
    student = User.objects.get(pk=student_id)
    student_settings: UserSettings = student.settings.first()

    if not student_settings.allow_video_training:
        return HttpResponseForbidden(
            "This student does not allow the usage of their videos for training."
        )
    if gesture_attempt.added_to_dataset:
        return HttpResponseForbidden(
            "This video has already been added to the dataset."
        )
    gesture = gesture_attempt.gesture
    attempts_video_path = (
        settings.USER_GESTURES_ROOT
        / str(student_id)
        / str(unit_attempt.unit.id)
        / str(unit_attempt.id)
        / f"{gesture.id}_{gesture_attempt.attempt}.webm"
    )
    new_gesture_path = (
        gesture.videos_location
        / f"{student_id}_{gesture_attempts_id}_{attempts_video_path.name}"
    )

    stream = ffmpeg.input(str(attempts_video_path))
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, str(new_gesture_path))
    ffmpeg.run(stream)

    gesture.status = Gesture.Status.PENDING
    gesture.save()

    gesture_attempt.added_to_dataset = True
    gesture_attempt.save()
    return JsonResponse({"status": "ok"})


@login_required
@teacher_or_admin_required
def manage_courses_view(request):
    """Page for teachers with an overview of their courses."""
    teacher = get_user(request)
    context = {
        "current_section": "manage_courses",
        "courses": Course.objects.filter(Q(creator=teacher)).all(),
    }
    return render(
        request,
        "sign_language_app/profile/classroom/courses/manage_courses.html",
        context,
    )


@login_required
@teacher_or_admin_required
def new_course_view(request):
    """Page for teacher to create new courses and units."""
    user = get_user(request)
    upload_gesture_form = UploadGestureForm(request.GET)

    if request.method == "POST":
        data_from_post = json.load(request)
        title = None
        description = None
        visibility = None
        difficulty = None
        new_units = []
        for entry in data_from_post:
            name = entry.get("name")
            value = entry.get("value")
            if name == "title":
                title = value
            elif name == "description":
                description = value
            elif name == "visibility":
                visibility = value
            elif name == "difficulty":
                difficulty = value
            elif name == "units":
                for i, unit in enumerate(value):
                    unit_name = unit["title"]
                    gesture_ids = unit["gesture_ids"]
                    gestures = Gesture.objects.filter(Q(id__in=gesture_ids))
                    unit = Unit(name=unit_name, ordering_number=i)
                    new_units.append((unit, gestures))

        if title and description and visibility and difficulty and new_units:
            new_course = Course(
                name=title,
                description=description,
                visibility=visibility,
                difficulty=difficulty,
                creator=user,
            )
            new_course.save()
            for i, (unit, gestures) in enumerate(new_units):
                unit.course = new_course
                unit.save()
                unit.gestures.set(gestures)
            return JsonResponse({})
        else:
            # messages.error(request, 'The form was not valid. Please try again and make sure to fill in all the necessary information')
            response = JsonResponse(
                {"error": "Some data was missing. Make sure you will in all the field."}
            )
            response.status_code = 403
            return response
    else:
        form = NewCourseForm()

    context = {
        "current_section": "manage_courses",
        "form": form,
        "upload_gesture_form": upload_gesture_form,
    }
    return render(
        request, "sign_language_app/profile/classroom/courses/new_course.html", context
    )


@login_required
@teacher_or_admin_required
def manage_gestures(request):
    """A view for teachers where they can add and view their own gestures."""
    user = get_user(request)
    upload_gesture_form = UploadGestureForm(request.GET)
    context = {
        "current_section": "manage_gestures",
        "upload_gesture_form": upload_gesture_form,
        "gestures": Gesture.objects.filter(Q(creator=user)).all(),
    }
    return render(
        request, "sign_language_app/profile/classroom/manage_gestures.html", context
    )


@login_required
def manage_teachers_view(request):
    """A view for student where they can manage who their teachers are to gain access to school courses."""
    student = get_user(request)
    if request.method == "POST":
        teacher_code_form = TeacherCodeForm(request.POST)
        if teacher_code_form.is_valid():
            teacher_code_string = teacher_code_form.cleaned_data.get("teacher_code")
            teacher_code = TeacherCode.objects.filter(code=teacher_code_string).first()
            if not teacher_code:
                messages.error(
                    request, "This teacher code is not valid or it has been removed."
                )
            else:
                if StudentsAccess.objects.filter(
                    student=student, teacher=teacher_code.teacher
                ).exists():
                    messages.error(
                        request,
                        f"You have already added this code for a teacher named {teacher_code.teacher.username}.",
                    )
                else:
                    student_access = StudentsAccess(
                        student=student, teacher=teacher_code.teacher
                    )
                    student_access.save()

    context = {
        "current_section": "manage_teachers",
        "teacher_code_form": TeacherCodeForm(),
        "teachers": StudentsAccess.get_teachers(student=student),
    }
    return render(
        request, "sign_language_app/profile/classroom/manage_teachers.html", context
    )


@login_required
@teacher_or_admin_required
def create_gesture(request):
    """Function that creates a new gestures from a list of uploaded videos."""
    user = get_user(request)
    if request.method == "POST":
        form = UploadGestureForm(request.POST, request.FILES)
        if form.is_valid():
            gesture_word = form.cleaned_data.get("word")
            left_hand = form.cleaned_data.get("left_hand")
            right_hand = form.cleaned_data.get("right_hand")
            existing_gesture = Gesture.objects.filter(
                Q(word__iexact=gesture_word) & Q(creator=user)
            ).first()
            if existing_gesture:
                messages.error(request, "You already created a gesture for that word.")
                return redirect("manage_gestures")
            if not left_hand and not right_hand:
                messages.error(
                    request,
                    "Please specify which hands are used to perform the gesture",
                )
                return redirect("manage_gestures")
            form.handle_uploaded_files(request, user=user)
            new_gesture = Gesture(
                word=gesture_word,
                left_hand=left_hand,
                right_hand=right_hand,
                creator=user,
                status=Gesture.Status.PENDING,
            )
            new_gesture.save()
        else:
            messages.error(
                request,
                "The form was not valid. Please try again and make sure to fill in all the necessary information",
            )
    return redirect("manage_gestures")


@login_required
@teacher_or_admin_required
def delete_gesture(request, gesture_id):
    user = get_user(request)
    gesture = get_object_or_404(Gesture, pk=gesture_id)
    if gesture.creator != user:
        messages.error(request, "You cannot delete a gesture that you didn't create.")
    messages.success(request, f"{gesture.word} has been deleted.")
    gesture.status = Gesture.Status.DELETED
    gesture.save()
    return redirect("manage_gestures")


@login_required
@teacher_or_admin_required
def delete_course(request, course_id):
    user = get_user(request)
    course = get_object_or_404(Course, pk=course_id)
    if course.creator != user:
        messages.error(request, "You cannot delete a course that you didn't create.")
    messages.success(request, f"{course.name} has been deleted.")
    course.delete()
    return redirect("manage_courses")


@login_required
@teacher_or_admin_required
def feedback(request):
    user = get_user(request)
    context = {"current_section": "feedback"}
    return render(request, "sign_language_app/profile/feedback.html", context)
