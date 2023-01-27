from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from learning_site import settings

handler404 = "sign_language_app.views.views.error_view"

urlpatterns = [
    path("", include("sign_language_app.urls")),
    path("admin/", admin.site.urls),
]

# WARNING: Django should not be used to server static and media files in production.
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
