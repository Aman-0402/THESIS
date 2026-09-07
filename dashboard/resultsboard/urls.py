from django.urls import path
from . import views

app_name = "resultsboard"
urlpatterns = [
    path("", views.overview, name="overview"),
]
