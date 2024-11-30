from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_search, name='upload_and_search'),
]
