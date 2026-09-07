from django.urls import path
from . import views

app_name = "resultsboard"
urlpatterns = [
    path("", views.overview, name="overview"),
    path("models/", views.model_comparison, name="model_comparison"),
    path("models/<str:model_name>/", views.model_detail, name="model_detail"),
]
