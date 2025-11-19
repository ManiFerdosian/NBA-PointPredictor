FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY assets/ ./assets/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data models

# Run data pipeline to create database
RUN python -m src.data_pipeline.load_nba_data

# Train the model
RUN python -m src.ml.train_model

# Expose port
EXPOSE 8080

# Set environment variables
ENV API_PORT=8080
ENV DB_PATH=data/db.nba.sqlite
ENV MODEL_PATH=models/nba_over20_model.pt

# Start the API server
# Use shell form to allow environment variable expansion
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${API_PORT:-8080}"

