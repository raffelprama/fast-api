# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create cache directory and set permissions for HuggingFace
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Set environment variables
ENV HF_HOME="/app/.cache"
# ENV TRANSFORMERS_CACHE="/app/.cache"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/

# Debugging: Check if main.py exists and print directory structure
RUN ls -la /app

# Expose the required port for FastAPI
EXPOSE 8000

# Set environment variables for the application
ENV LOCUST_HOST=https://dekallm.cloudeka.ai
ENV LOCUST_USERS=100
ENV LOCUST_SPAWN_RATE=100
ENV LOCUST_DURATION=60
ENV LOCUST_MODEL=deepseek-ai/DeepSeek-R1-70B
ENV LOCUST_TOKENIZER=deepseek-ai/DeepSeek-R1
ENV LOCUST_DATASET=mteb/banking77


# Run FastAPI using Uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
