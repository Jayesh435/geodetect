#!/bin/bash

# Run database migrations
python manage.py migrate

# Create media directories if they don't exist
mkdir -p /app/media/uploads /app/media/processed

# Start the application
exec gunicorn geodetect.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 120