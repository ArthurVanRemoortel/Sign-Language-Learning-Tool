from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_test', views.video_test, name='video_test'),
]