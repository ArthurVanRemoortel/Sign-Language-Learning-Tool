from django.urls import path, include
from rest_framework import routers
from .views import views
from .views import profile_views
from .views import auth as auth_views
from .views import api as api_views
from django.contrib.auth import views as contrib_auth_views
from sign_language_app import forms

API_PREFIX = "api/"

router = routers.DefaultRouter()
router.register("gestures", api_views.GestureViewSet, basename="api-gestures")

# general patters
urlpatterns = [
    path(API_PREFIX, include(router.urls)),
    path("", views.index, name="index"),
    path("courses", views.courses_overview, name="courses_overview"),
    path("unit/<int:unit_id>", views.unit_view, name="unit"),
    path(
        "unit/<int:unit_id>/summary/<int:attempt_id>",
        views.unit_summary,
        name="unit_summary",
    ),
    path("unit/<int:unit_id>/save", views.save_unit_attempt, name="save_unit_attempt"),
    path(
        "api/upload_gesture_video",
        views.upload_gesture_video,
        name="upload_gesture_video",
    ),
]

# Profile Patterns
urlpatterns += [
    path("profile", profile_views.profile_overview, name="profile"),
    path("profile_settings", profile_views.profile_settings, name="profile_settings"),
    path("manage_students", profile_views.manage_students_view, name="manage_students"),
    path("manage_courses", profile_views.manage_courses_view, name="manage_courses"),
    path("manage_gestures", profile_views.manage_gestures, name="manage_gestures"),
    path(
        "manage_gestures/<int:gesture_id>/delete",
        profile_views.delete_gesture,
        name="delete_gesture",
    ),
    path(
        "manage_courses/<int:course_id>/delete",
        profile_views.delete_course,
        name="delete_course",
    ),
    path("manage_teachers", profile_views.manage_teachers_view, name="manage_teachers"),
    path("feedback", profile_views.feedback, name="feedback"),
    path("create_gesture", profile_views.create_gesture, name="create_gesture"),
    path("manage_courses", profile_views.manage_courses_view, name="classroom"),
    path("manage_courses/new_course", profile_views.new_course_view, name="new_course"),
    path(
        "manage_students/regenerate_teacher_code",
        profile_views.regenerate_teacher_code,
        name="regenerate_teacher_code",
    ),
    path(
        "manage_students/<int:student_id>/remove",
        profile_views.remove_student_from_classroom,
        name="remove_student_from_classroom",
    ),
    path(
        "manage_teachers/<int:teacher_id>/remove_teacher",
        profile_views.remove_teacher,
        name="remove_teacher",
    ),
]

# Authentication patterns
urlpatterns += [
    path(
        "logout",
        contrib_auth_views.LogoutView.as_view(
            template_name="sign_language_app/auth/logout.html"
        ),
        name="logout",
    ),
    path("register", auth_views.register_account, name="register"),
    path("login", auth_views.login_account, name="login"),
    path(
        "password_reset",
        contrib_auth_views.PasswordResetView.as_view(
            template_name="sign_language_app/auth/password_reset.html",
            form_class=forms.ResetPasswordAuthenticationForm,
        ),
        name="password_reset",
    ),
    path(
        "password_reset_confirm/<uidb64>/<token>/",
        contrib_auth_views.PasswordResetConfirmView.as_view(
            template_name="sign_language_app/auth/password_reset_confirm.html",
            form_class=forms.NewPasswordAuthenticationForm,
        ),
        name="password_reset_confirm",
    ),
    path(
        "password_reset/done",
        contrib_auth_views.PasswordResetDoneView.as_view(
            template_name="sign_language_app/auth/password_reset_done.html"
        ),
        name="password_reset_done",
    ),
    path(
        "password_reset/complete",
        contrib_auth_views.PasswordResetCompleteView.as_view(
            template_name="sign_language_app/auth/password_reset_complete.html"
        ),
        name="password_reset_complete",
    ),
]

urlpatterns += [
    path(
        "api/test", api_views.test_auth, name="check_user_input"
    ),  # TODO: Fix this path
    path("api/retrain_model", api_views.trigger_retrain_model),
]
