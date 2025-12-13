# Legal Text Decoder - Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/

# Create directories INSIDE the container
RUN mkdir -p output/models log data/raw data/processed

# Make run script executable
RUN chmod +x src/run.sh

# Default command
CMD ["/bin/bash", "/app/src/run.sh"]