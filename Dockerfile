FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps for opencv-python-headless and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Run database migrations
RUN python manage.py migrate

# Create media directories
RUN mkdir -p /app/media/uploads /app/media/processed

EXPOSE 8000

CMD ["gunicorn", "geodetect.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
