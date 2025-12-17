from django.urls import path
from .views import home, detect_text

urlpatterns = [
    path("", home, name="home"),                # fast page load
    path("api/detect/", detect_text, name="detect_text"),
]
