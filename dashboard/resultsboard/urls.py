from django.urls import path
from . import views

app_name = "resultsboard"
urlpatterns = [
    path("", views.overview, name="overview"),
    path("models/", views.model_comparison, name="model_comparison"),
]
