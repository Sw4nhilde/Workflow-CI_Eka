FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install mlflow==3.13.0 scikit-learn==1.9.0 pandas cloudpickle

# Create directory for model
WORKDIR /app

# Copy model
COPY model /app/model

# Set environment variables
ENV MLFLOW_MODEL_PATH=/app/model

# Expose port
EXPOSE 8080

# Run MLflow server
CMD mlflow models serve -m /app/model -h 0.0.0.0 -p 8080 --no-conda
