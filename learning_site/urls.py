from django.contrib import admin
from django.urls import path, include


handler404 = 'sign_language_app.views.views.error_view'

urlpatterns = [
    path("", include("sign_language_app.urls")),
    path("admin/", admin.site.urls),
]