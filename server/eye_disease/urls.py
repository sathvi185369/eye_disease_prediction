from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_eye_disease, name='predict_eye_disease'),
]