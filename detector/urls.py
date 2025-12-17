from django.urls import path
from .views import home, detect_text

urlpatterns = [
    path("", home),
    path("api/detect/", detect_text),
]
