import json
import os
import uuid
import mimetypes

from django.conf import settings
from django.http import JsonResponse, FileResponse, Http404
from django.shortcuts import render, get_object_or_404

from .forms import ImageUploadForm
from .models import ImageUpload
from .utils import extract_gps, detect_objects, analyze_with_gemini


def upload_image(request):
    """Handle image upload form display and submission."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.save()

            original_path = instance.original_image.path

            # ── Extract GPS ──
            try:
                # First try GPS coordinates from camera/geolocation API
                latitude_param = request.POST.get('latitude')
                longitude_param = request.POST.get('longitude')
                
                if latitude_param and longitude_param:
                    try:
                        lat = round(float(latitude_param), 6)
                        lon = round(float(longitude_param), 6)
                        # Validate coordinates are sane
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            instance.latitude = lat
                            instance.longitude = lon
                            print(f"GPS from geolocation API: {lat}, {lon}")
                        else:
                            raise ValueError("Invalid coordinate range")
                    except (ValueError, TypeError):
                        print(f"Invalid GPS coords from frontend: {latitude_param}, {longitude_param}")
                        lat, lon = extract_gps(original_path)
                        instance.latitude = lat
                        instance.longitude = lon
                else:
                    # Fallback to image-based GPS extraction
                    lat, lon = extract_gps(original_path)
                    instance.latitude = lat
                    instance.longitude = lon
            except Exception:
                instance.latitude = None
                instance.longitude = None

            # ── Run tree detection ──
            try:
                ext = os.path.splitext(original_path)[1]
                processed_filename = f"processed_{uuid.uuid4().hex}{ext}"
                processed_rel = os.path.join('processed', processed_filename)
                processed_abs = os.path.join(settings.MEDIA_ROOT, processed_rel)

                detections = detect_objects(original_path, processed_abs)

                instance.processed_image.name = processed_rel
                instance.detections_json = detections
            except Exception as e:
                import traceback
                traceback.print_exc()
                instance.detections_json = [{'error': str(e)}]

            # ── Run Gemini AI analysis ──
            try:
                description = analyze_with_gemini(original_path)
                instance.gemini_description = description
            except Exception as e:
                import traceback
                traceback.print_exc()
                instance.gemini_description = f"AI analysis failed: {str(e)}"

            instance.save()

            return render(request, 'detector/result.html', {
                'instance': instance,
                'detections': instance.detections_json,
                'detections_json': json.dumps(instance.detections_json, indent=2),
            })
    else:
        form = ImageUploadForm()

    return render(request, 'detector/upload.html', {'form': form})


def result_json(request, pk):
    """Return detection results as JSON."""
    instance = get_object_or_404(ImageUpload, pk=pk)
    return JsonResponse({
        'id': instance.pk,
        'latitude': instance.latitude,
        'longitude': instance.longitude,
        'detections': instance.detections_json,
        'original_image': instance.original_image.url if instance.original_image else None,
        'processed_image': instance.processed_image.url if instance.processed_image else None,
    })


def download_processed(request, pk):
    """Download the processed image."""
    instance = get_object_or_404(ImageUpload, pk=pk)
    if not instance.processed_image:
        raise Http404("No processed image available.")
    file_path = instance.processed_image.path
    if not os.path.exists(file_path):
        raise Http404("Processed image file not found.")
    content_type, _ = mimetypes.guess_type(file_path)
    response = FileResponse(
        open(file_path, 'rb'),
        content_type=content_type or 'image/jpeg',
    )
    filename = f"geodetect_{instance.pk}_processed{os.path.splitext(file_path)[1]}"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response
