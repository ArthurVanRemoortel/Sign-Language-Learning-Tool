from django.urls import path, include
from .views import views
from .views import auth as auth_views
from .views import api as api_views
from django.contrib.auth import views as contrib_auth_views
from sign_language_app import forms

urlpatterns = [
    path('', views.index, name='index'),
    path('courses', views.courses_overview, name='courses_overview'),
    path('unit/<int:unit_id>', views.unit_view, name='unit'),
]

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