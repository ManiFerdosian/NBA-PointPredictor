#!/usr/bin/env bash

# Build and run the NBA Over/Under API container

echo "Building Docker image..."
docker build -t nba-overunder-api:latest .

echo "Running container..."
docker run --rm -p 8080:8080 --env-file .env nba-overunder-api:latest

