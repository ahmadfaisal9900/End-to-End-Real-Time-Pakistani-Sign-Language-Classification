FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create directory for model mounting
RUN mkdir -p /app/models

# Copy application code
COPY . .

# Set environment variable for model path
ENV MODEL_DIR=/app/models
ENV DOCKER_CONTAINER=true

RUN pip install -e .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "run_api.py"]
