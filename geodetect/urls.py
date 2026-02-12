"""
URL configuration for geodetect project.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),
]

# Serve media files (uploads & processed images) in all environments
# On Render free tier there's no separate media CDN
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
