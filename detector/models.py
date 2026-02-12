from django.db import models
import uuid
import os


def upload_to(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    return os.path.join('uploads', filename)


def processed_to(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"processed_{uuid.uuid4().hex}.{ext}"
    return os.path.join('processed', filename)


class ImageUpload(models.Model):
    original_image = models.ImageField(upload_to=upload_to)
    processed_image = models.ImageField(upload_to=processed_to, blank=True, null=True)
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    detections_json = models.JSONField(default=list, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Image #{self.pk} - {self.uploaded_at:%Y-%m-%d %H:%M}"
