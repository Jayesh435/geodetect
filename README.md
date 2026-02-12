# GeoDetect — Django YOLOv8 Geotagged Image Detector

A Django web application that accepts geotagged image uploads, extracts GPS coordinates from EXIF metadata, runs YOLOv8 object detection, and returns annotated images with bounding boxes, class labels, confidence scores, and GPS data.

---

## Features

- **Image Upload** — Simple drag-and-drop or file picker interface
- **GPS Extraction** — Reads EXIF GPS DMS data and converts to decimal degrees
- **YOLOv8 Detection** — Uses `yolov8n.pt` (nano) for fast object detection
- **Bounding Boxes** — Drawn on image via OpenCV with class-colored labels
- **JSON API** — Full detection data available at `/api/result/<id>/`
- **Google Maps Link** — One-click link to view GPS location on the map
- **Error Handling** — Graceful handling of missing geotags, bad images, etc.

---

## Project Structure

```
cojag hackthon/
├── manage.py
├── requirements.txt
├── README.md
├── geodetect/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── detector/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   ├── utils.py          ← GPS extraction + YOLO detection logic
│   ├── views.py
│   └── templates/
│       └── detector/
│           ├── upload.html
│           └── result.html
└── media/                 ← Created automatically
    ├── uploads/
    └── processed/
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Migrations

```bash
python manage.py makemigrations detector
python manage.py migrate
```

### 3. Start the Server

```bash
python manage.py runserver
```

### 4. Open in Browser

Navigate to **http://127.0.0.1:8000/** and upload a geotagged image.

---

## API Endpoint

```
GET /api/result/<id>/
```

Returns JSON:
```json
{
  "id": 1,
  "latitude": 37.7749,
  "longitude": -122.4194,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.9234,
      "bbox": { "x1": 100, "y1": 50, "x2": 300, "y2": 400 }
    }
  ],
  "original_image": "/media/uploads/abc123.jpg",
  "processed_image": "/media/processed/processed_def456.jpg"
}
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Django 4.2+ |
| Detection | YOLOv8 (ultralytics) |
| Image I/O | Pillow, OpenCV |
| GPS Parse | Pillow EXIF + custom DMS→decimal |
| Model | yolov8n.pt (auto-downloaded) |

---

## Notes

- The YOLOv8 nano model (`yolov8n.pt`) is downloaded automatically on first run.
- Images without GPS EXIF data will still be processed for object detection — the GPS section will show "No GPS geotag found."
- Maximum upload size is 20 MB (configurable in `settings.py`).
