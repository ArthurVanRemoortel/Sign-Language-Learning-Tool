from django.urls import path, include
from .views import views
from .views import auth as auth_views
from .views import api as api_views
from django.contrib.auth import views as contrib_auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_test', views.video_test, name='video_test'),
    path('courses', views.courses_overview, name='courses_overview'),
    path('unit/<int:unit_id>', views.unit_view, name='unit'),
]

urlpatterns += [
    path('login', contrib_auth_views.LoginView.as_view(template_name='sign_language_app/auth/login.html'), name='login'),
    path('logout', contrib_auth_views.LogoutView.as_view(template_name='sign_language_app/auth/logout.html'), name='logout'),
    path('password_reset', contrib_auth_views.PasswordResetView.as_view(template_name='sign_language_app/auth/password_reset.html'), name='password_reset'),
    path('password_reset/done', contrib_auth_views.PasswordResetDoneView.as_view(template_name='sign_language_app/auth/password_reset_done.html'), name='password_reset_done'),
    path('register', auth_views.register_account, name='register'),
]

urlpatterns += [
    path('api/test', api_views.test_auth)
]