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
    path('', views.index, name='index'),
    path('courses', views.courses_overview, name='courses_overview'),
    path('unit/<int:unit_id>', views.unit_view, name='unit'),
]

# Profile Patterns
urlpatterns += [
    path('profile', profile_views.profile_overview, name='profile'),
    path('profile_settings', profile_views.profile_settings, name='profile_settings'),
    path('manage_students', profile_views.manage_students, name='manage_students'),
    path('manage_courses', profile_views.manage_courses_view, name='manage_courses'),
    path('manage_gestures', profile_views.manage_gestures, name='manage_gestures'),
    path('feedback', profile_views.feedback, name='feedback'),
    path('create_gesture', profile_views.create_gesture, name='create_gesture'),
    path('manage_courses', profile_views.manage_courses_view, name='classroom'),
    path('manage_courses/new_course', profile_views.new_course_view, name='new_course'),
]

# Authentication patterns
urlpatterns += [
    path('logout', contrib_auth_views.LogoutView.as_view(template_name='sign_language_app/auth/logout.html'), name='logout'),
    path('register', auth_views.register_account, name='register'),
    path('login', auth_views.login_account, name='login'),

    path('password_reset', contrib_auth_views.PasswordResetView.as_view(template_name='sign_language_app/auth/password_reset.html', form_class=forms.ResetPasswordAuthenticationForm), name='password_reset'),
    path('password_reset_confirm/<uidb64>/<token>/', contrib_auth_views.PasswordResetConfirmView.as_view(template_name='sign_language_app/auth/password_reset_confirm.html', form_class=forms.NewPasswordAuthenticationForm), name='password_reset_confirm'),
    path('password_reset/done', contrib_auth_views.PasswordResetDoneView.as_view(template_name='sign_language_app/auth/password_reset_done.html'), name='password_reset_done'),
    path('password_reset/complete', contrib_auth_views.PasswordResetCompleteView.as_view(template_name='sign_language_app/auth/password_reset_complete.html'), name='password_reset_complete'),
]

urlpatterns += [
    path('api/test', api_views.test_auth)
]