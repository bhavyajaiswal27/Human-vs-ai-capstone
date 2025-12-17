from django.urls import path
from .views import detect_text

urlpatterns = [
    path("", detect_text, name="home"),      
    path("api/detect/", detect_text, name="detect_text"),
]
