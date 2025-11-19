# Containerized NBA Points Over/Under Predictor

A complete end-to-end machine learning system that predicts whether an NBA player will score over or under a chosen points line in a single game. The system uses SQL for data storage, PyTorch for machine learning, and FastAPI for the REST API, all running in a Docker container.

## Executive Summary

This project implements a binary classification model to predict NBA player performance relative to a points line. The system processes historical NBA box score data, trains a neural network model, and exposes predictions through a RESTful API. The entire application runs in a single Docker container, making it easy to deploy and test.

**Problem**: Predict whether an NBA player will score over or under a specified points line in a game.

**Solution**: A containerized ML system with:
- SQLite database for historical game data
- PyTorch neural network for predictions
- FastAPI REST API for inference
- Complete Docker containerization

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                          │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Data       │    │   Training   │    │   REST API   │  │
│  │  Pipeline    │───▶│   Script     │───▶│   (FastAPI)  │  │
│  │              │    │              │    │              │  │
│  │ CSV → SQLite │    │ PyTorch Model│    │ /predict     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         └────────────────────┴────────────────────┘          │
│                              │                               │
│                    ┌─────────▼─────────┐                    │
│                    │   SQLite Database │                    │
│                    │  (db.nba.sqlite)  │                    │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Data Pipeline** (`src/data_pipeline/load_nba_data.py`)
   - Loads CSV data from `assets/nba_player_games_sample.csv`
   - Cleans and validates data
   - Creates SQLite database with `player_games` table
   - Generates binary labels (over_20: 1 if points >= 20, else 0)

2. **ML Model** (`src/ml/`)
   - Feedforward neural network: Linear → ReLU → Linear → Sigmoid
   - Binary cross-entropy loss, Adam optimizer
   - Trained on 6 features: minutes, rebounds, assists, field_goals_attempted, three_pa, free_throws_attempted
   - Saves trained weights to `models/nba_over20_model.pt`

3. **REST API** (`src/api/main.py`)
   - FastAPI application with Uvicorn server
   - Endpoints: `/health`, `/predict`, `/example`
   - Accepts feature vectors and points line, returns probability and prediction

## Quick Start

### Prerequisites

- Docker installed and running
- Git (optional, for cloning)

### Running with Docker

1. **Clone or navigate to the project directory:**
   ```bash
   cd nba-overunder-api
   ```

2. **Create a `.env` file** (copy from `.env.example`):
   ```bash
   API_PORT=8080
   DB_PATH=data/db.nba.sqlite
   MODEL_PATH=models/nba_over20_model.pt
   LOG_LEVEL=info
   ```

3. **Build and run using the provided script:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

   Or manually:
   ```bash
   docker build -t nba-overunder-api:latest .
   docker run --rm -p 8080:8080 --env-file .env nba-overunder-api:latest
   ```

4. **Test the API:**
   ```bash
   # Health check
   curl http://localhost:8080/health
   
   # Get example request
   curl http://localhost:8080/example
   
   # Make a prediction
   curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{
       "player_name": "LeBron James",
       "points_line": 20,
       "features": {
         "minutes": 34.5,
         "rebounds": 8.0,
         "assists": 7.2,
         "field_goals_attempted": 18,
         "three_pa": 6,
         "free_throws_attempted": 6
       }
     }'
   ```

### Running Tests

Run pytest tests locally (requires Python environment):

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## API Endpoints

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

### POST `/predict`

Predict whether a player will go over or under the points line.

**Request:**
```json
{
  "player_name": "LeBron James",
  "points_line": 20,
  "features": {
    "minutes": 34.5,
    "rebounds": 8.0,
    "assists": 7.2,
    "field_goals_attempted": 18,
    "three_pa": 6,
    "free_throws_attempted": 6
  }
}
```

**Response:**
```json
{
  "prob_over": 0.73,
  "prediction": "over",
  "points_line": 20
}
```

### GET `/example`

Returns an example request body for the `/predict` endpoint.

## Design Decisions

1. **SQLite over PostgreSQL**: Chosen for simplicity and single-file deployment. Perfect for this small-scale project.

2. **Fixed 20-point training line**: The model is trained on a binary classification task (over/under 20 points). For inference with different lines, we adjust the prediction threshold heuristically.

3. **Simple neural network architecture**: A small feedforward network (6 inputs → 32 hidden → 1 output) is sufficient for this tabular data problem and trains quickly on CPU.

4. **Feature vector in request**: For simplicity, features are passed directly in the API request rather than looking up player statistics. This keeps the API stateless and fast.

5. **Docker build-time training**: The model is trained during the Docker build process, ensuring the container always has a trained model ready. This trades flexibility for reproducibility.

6. **FastAPI over Flask**: FastAPI provides automatic OpenAPI documentation, better type validation with Pydantic, and async support out of the box.

## Project Structure

```
nba-overunder-api/
├── src/
│   ├── data_pipeline/
│   │   └── load_nba_data.py      # CSV to SQLite pipeline
│   ├── ml/
│   │   ├── model.py               # PyTorch model definition
│   │   └── train_model.py         # Training script
│   └── api/
│       └── main.py                # FastAPI application
├── assets/
│   └── nba_player_games_sample.csv  # Sample NBA data
├── tests/
│   └── test_api.py                # Pytest smoke tests
├── models/                         # Trained model storage
├── data/                           # SQLite database storage
├── Dockerfile                      # Container definition
├── requirements.txt               # Python dependencies
├── .env.example                    # Environment variable template
├── run.sh                          # One-liner launcher
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Environment Variables

- `API_PORT`: Port for the FastAPI server (default: 8080)
- `DB_PATH`: Path to SQLite database file (default: data/db.nba.sqlite)
- `MODEL_PATH`: Path to trained PyTorch model (default: models/nba_over20_model.pt)
- `LOG_LEVEL`: Logging level (default: info)

## Cloud Deployment

The application is designed to be cloud-ready:

- Listens on `0.0.0.0` (all interfaces)
- Port configurable via environment variable
- No hardcoded local paths
- Stateless API design

**Compatible with:**
- Azure Web App for Containers
- Render
- Railway
- Any container platform

## Course Concepts Demonstrated

✅ **SQL and relational data**: SQLite database with structured player_games table  
✅ **Scripting and data pipelines**: Automated CSV to SQL pipeline  
✅ **Containerization with Docker**: Complete Docker setup with single-command run  
✅ **Optional logging and metrics**: Health check endpoint with status information  

## License

MIT License - see LICENSE file for details.

