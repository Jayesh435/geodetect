"""
Utility functions for GPS extraction, HSV-based
green-vegetation / tree detection, and Gemini AI image analysis.
"""

import cv2
import re
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
from django.conf import settings
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)


# ─── GPS EXTRACTION ───────────────────────────────────────────────

def _dms_to_decimal(dms, ref):
    """Convert GPS DMS to decimal degrees."""
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ('S', 'W'):
            decimal = -decimal
        return round(decimal, 6)
    except (TypeError, IndexError, ZeroDivisionError):
        return None


def _extract_gps_from_exif(image_path):
    """Extract GPS latitude/longitude from EXIF data."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None, None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == 'GPSInfo':
                if isinstance(value, dict):
                    for gps_tag_id, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag_name] = gps_value
                else:
                    return None, None

        if not gps_info:
            return None, None

        lat_dms = gps_info.get('GPSLatitude')
        lat_ref = gps_info.get('GPSLatitudeRef')
        lon_dms = gps_info.get('GPSLongitude')
        lon_ref = gps_info.get('GPSLongitudeRef')

        if not all([lat_dms, lat_ref, lon_dms, lon_ref]):
            return None, None

        latitude = _dms_to_decimal(lat_dms, lat_ref)
        longitude = _dms_to_decimal(lon_dms, lon_ref)
        return latitude, longitude
    except Exception as e:
        logger.warning(f"EXIF GPS extraction failed: {e}")
        return None, None


def _extract_gps_from_xmp(image_path):
    """Try extracting GPS from XMP metadata embedded in the file."""
    try:
        with open(image_path, 'rb') as f:
            data = f.read()

        xmp_start = data.find(b'<x:xmpmeta')
        xmp_end = data.find(b'</x:xmpmeta')
        if xmp_start == -1 or xmp_end == -1:
            return None, None

        xmp_str = data[xmp_start:xmp_end + 12].decode('utf-8', errors='ignore')

        lat_match = re.search(r'exif:GPSLatitude="([^"]+)"', xmp_str)
        lon_match = re.search(r'exif:GPSLongitude="([^"]+)"', xmp_str)

        if lat_match and lon_match:
            lat = _parse_xmp_gps(lat_match.group(1))
            lon = _parse_xmp_gps(lon_match.group(1))
            if lat is not None and lon is not None:
                return lat, lon

        return None, None
    except Exception as e:
        logger.warning(f"XMP GPS extraction failed: {e}")
        return None, None


def _parse_xmp_gps(value):
    """Parse XMP GPS string like '19,30.9369N' to decimal degrees."""
    try:
        match = re.match(r'(\d+),(\d+\.?\d*)(.*?)([NSEW])', value)
        if match:
            deg = float(match.group(1))
            minutes = float(match.group(2))
            ref = match.group(4)
            decimal = deg + minutes / 60.0
            if ref in ('S', 'W'):
                decimal = -decimal
            return round(decimal, 6)
    except Exception:
        pass
    return None


def _extract_gps_from_text_overlay(image_path):
    """
    Fallback: scan image file bytes for patterns like 'Lat 19.515616' and 'Long 79.966319'.
    GPS Map Camera and similar apps embed this text as metadata strings in the file.
    """
    try:
        with open(image_path, 'rb') as f:
            raw = f.read()

        text = raw.decode('utf-8', errors='ignore')

        pattern = r'Lat[:\s]*(-?\d+\.?\d*)[°]?\s*Long[:\s]*(-?\d+\.?\d*)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            lat = round(float(match.group(1)), 6)
            lon = round(float(match.group(2)), 6)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon

        pattern2 = r'latitude[:\s]*(-?\d+\.?\d*)[°,\s]+longitude[:\s]*(-?\d+\.?\d*)'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            lat = round(float(match2.group(1)), 6)
            lon = round(float(match2.group(2)), 6)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon

        return None, None
    except Exception as e:
        logger.warning(f"Text overlay GPS extraction failed: {e}")
        return None, None


def extract_gps(image_path):
    """
    Extract GPS coordinates using multiple strategies:
    1. Standard EXIF GPS tags
    2. XMP metadata
    3. Text overlay / embedded strings (GPS Map Camera apps)
    """
    lat, lon = _extract_gps_from_exif(image_path)
    if lat is not None and lon is not None:
        logger.info(f"GPS from EXIF: {lat}, {lon}")
        return lat, lon

    lat, lon = _extract_gps_from_xmp(image_path)
    if lat is not None and lon is not None:
        logger.info(f"GPS from XMP: {lat}, {lon}")
        return lat, lon

    lat, lon = _extract_gps_from_text_overlay(image_path)
    if lat is not None and lon is not None:
        logger.info(f"GPS from text overlay: {lat}, {lon}")
        return lat, lon

    return None, None


# ─── GEMINI AI IMAGE ANALYSIS ──────────────────────────────────────────

def analyze_with_gemini(image_path, lat=None, lon=None):
    """
    Analyze image using Google Gemini AI for comprehensive description.
    Returns detailed analysis of the image content, objects, environment, etc.
    """
    try:
        # Configure Gemini API
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - skipping AI analysis")
            return "AI analysis unavailable - API key not configured"
        
        genai.configure(api_key=api_key)
        
        # Load the image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create the model
        # Use Gemini 2.5 Flash for best balance of speed and vision accuracy
        model_name = 'gemini-2.5-flash'
        try:
            model = genai.GenerativeModel(model_name)
        except Exception:
             try:
                # Fallbacks in case of deployment rollouts
                model = genai.GenerativeModel('gemini-2.0-flash')
             except:
                model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the image
        image_parts = [{
            "mime_type": "image/jpeg",
            "data": image_data
        }]
        
        # Build prompt with location context if available
        location_context = ""
        if lat and lon:
            location_context = f"\n**Location Context**: This image was taken at approx Latitude {lat}, Longitude {lon}. Use this to help identify regional vegetation types."

        # Create comprehensive prompt for analysis
        prompt = f"""Act as an expert botanist and geospatial analyst. Analyze this image.{location_context}

1.  **Summary of Key Objects**:
    *   Provide a concise list of the main objects visible (e.g., "2 large Banyan trees, 1 light pole, brick wall").

2.  **Flora Identification**:
    *   Identify the main tree/plant species (Common & Scientific Name).
    *   Briefly note health/condition.

3.  **Environment**:
    *   One sentence describing the location/setting.

Keep the response short, direct, and focused on identifying the main subjects."""

        # Generate response
        response = model.generate_content([prompt] + image_parts)
        
        if response and response.text:
            logger.info("Gemini analysis completed successfully")
            return response.text.strip()
        else:
            logger.warning("Gemini returned empty response")
            return "AI analysis completed but no description generated"
            
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return f"AI analysis failed: {str(e)}"


# ─── HSV GREEN-VEGETATION DETECTION ──────────────────────────────

def _detect_green_vegetation(image_bgr, min_area_ratio=0.003, max_detections=20):
    """
    Detect individual green vegetation / tree canopies using HSV segmentation
    + distance-transform watershed to separate touching canopies.

    Returns list of dicts [{class_name, confidence, bbox}, ...]
    """
    h_img, w_img = image_bgr.shape[:2]
    img_area = h_img * w_img

    # ── resize for speed ──
    scale = 1.0
    if max(h_img, w_img) > 1024:
        scale = 1024 / max(h_img, w_img)
        small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr.copy()

    sh, sw = small.shape[:2]
    inv_scale = 1.0 / scale

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Medium green range (covers bright + dark foliage)
    mask = cv2.inRange(hsv, np.array([25, 30, 25]), np.array([90, 255, 250]))

    # Close to fill leaf gaps
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Distance transform to find individual tree cores
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist < 5:
        return []  # no meaningful green region

    # Threshold at 30% of max distance → core markers per tree
    _, sure_fg = cv2.threshold(dist, 0.3 * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Connected components → one label per tree
    n_labels, labels = cv2.connectedComponents(sure_fg)

    min_px = sh * sw * min_area_ratio  # min pixels for a core
    detections = []

    for label_id in range(1, n_labels):
        pts = np.where(labels == label_id)
        if len(pts[0]) < min_px:
            continue

        # Bounding box in resized coords
        y1s, x1s = int(pts[0].min()), int(pts[1].min())
        y2s, x2s = int(pts[0].max()), int(pts[1].max())

        # Pad box by ~20% to cover full canopy (core is smaller)
        pad_x = int((x2s - x1s) * 0.20)
        pad_y = int((y2s - y1s) * 0.20)
        x1s = max(0, x1s - pad_x)
        y1s = max(0, y1s - pad_y)
        x2s = min(sw, x2s + pad_x)
        y2s = min(sh, y2s + pad_y)

        # Scale back to original image coords
        x1 = max(0, int(x1s * inv_scale))
        y1 = max(0, int(y1s * inv_scale))
        x2 = min(w_img, int(x2s * inv_scale))
        y2 = min(h_img, int(y2s * inv_scale))

        if x2 - x1 < 30 or y2 - y1 < 30:
            continue

        box_area = (x2 - x1) * (y2 - y1)
        if box_area > img_area * 0.40:
            continue  # skip oversized regions

        # Confidence = green pixel density in the ROI
        roi_hsv = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        roi_mask = cv2.inRange(roi_hsv, np.array([25, 30, 25]), np.array([90, 255, 250]))
        density = np.count_nonzero(roi_mask) / (roi_mask.shape[0] * roi_mask.shape[1] + 1e-6)
        confidence = round(min(density * 1.15, 0.99), 2)

        if confidence < 0.20:
            continue

        detections.append({
            'class_name': 'Tree',
            'confidence': confidence,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections[:max_detections]


def _nms_merge(boxes, iou_thresh=0.3):
    """Simple non-maximum suppression / merge for (x1,y1,x2,y2,area) tuples."""
    if not boxes:
        return []
    kept = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        bx1, by1, bx2, by2, _ = boxes[i]
        # Absorb smaller overlapping boxes
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            ox1, oy1, ox2, oy2, _ = boxes[j]
            inter_x1 = max(bx1, ox1)
            inter_y1 = max(by1, oy1)
            inter_x2 = min(bx2, ox2)
            inter_y2 = min(by2, oy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_j = (ox2 - ox1) * (oy2 - oy1)
            if area_j > 0 and inter_area / area_j > iou_thresh:
                # Expand current box to include j
                bx1 = min(bx1, ox1)
                by1 = min(by1, oy1)
                bx2 = max(bx2, ox2)
                by2 = max(by2, oy2)
                used[j] = True
        kept.append((bx1, by1, bx2, by2))
        used[i] = True
    return kept


# ─── OBJECT DETECTION (TREES ONLY) ───────────────────────────────

TREE_COLOR = (0, 200, 0)  # bright green for tree boxes


def detect_objects(image_path, output_path):
    """
    Detect trees using HSV green-vegetation segmentation.
    Draws bounding boxes on the image and saves to output_path.

    Returns:
        list of dicts: [{class_name, confidence, bbox: {x1,y1,x2,y2}}, ...]
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    detections = _detect_green_vegetation(img)
    logger.info(f"Tree detector found {len(detections)} tree(s)")

    for det in detections:
        bx = det['bbox']
        cv2.rectangle(img, (bx['x1'], bx['y1']), (bx['x2'], bx['y2']), TREE_COLOR, 3)
        label = f"Tree {det['confidence']:.2f}"
        _draw_label(img, label, bx['x1'], bx['y1'], TREE_COLOR)

    # Save processed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), img)

    return detections


def _draw_label(img, label, x1, y1, color, font_scale=0.7, thickness=2):
    """Draw a text label with a filled background above the bounding box."""
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
