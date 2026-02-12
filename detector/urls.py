from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.upload_image, name='upload'),
    path('api/result/<int:pk>/', views.result_json, name='result_json'),
    path('download/<int:pk>/', views.download_processed, name='download'),
]
