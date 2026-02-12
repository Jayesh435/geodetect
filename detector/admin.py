from django.contrib import admin
from .models import ImageUpload


@admin.register(ImageUpload)
class ImageUploadAdmin(admin.ModelAdmin):
    list_display = ('pk', 'latitude', 'longitude', 'uploaded_at')
    readonly_fields = ('detections_json',)
