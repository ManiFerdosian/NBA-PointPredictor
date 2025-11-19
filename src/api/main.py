"""
FastAPI application for NBA over/under prediction.
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from src.ml.model import NBAOverUnderModel


# Load environment variables
DB_PATH = os.getenv("DB_PATH", "data/db.nba.sqlite")
MODEL_PATH = os.getenv("MODEL_PATH", "models/nba_over20_model.pt")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Initialize FastAPI app
app = FastAPI(title="NBA Over/Under Predictor API")

# Global model variable
model = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    player_name: str
    points_line: float
    features: dict


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prob_over: float
    prediction: str
    points_line: float


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model
    
    # Get absolute path to model
    project_root = Path(__file__).parent.parent.parent
    model_file = project_root / MODEL_PATH
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Load model
    print(f"Loading model from {model_file}...")
    model = NBAOverUnderModel(input_size=6, hidden_size=32)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")
    
    # Verify database connection
    db_file = project_root / DB_PATH
    if db_file.exists():
        print(f"Database found at {db_file}")
    else:
        print(f"Warning: Database not found at {db_file}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    
    # Check database path
    project_root = Path(__file__).parent.parent.parent
    db_file = project_root / DB_PATH
    db_exists = db_file.exists()
    
    response = {
        "status": "ok",
        "model_loaded": model is not None,
        "db_connected": db_exists
    }
    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict whether a player will go over or under the points line.
    
    The model was trained on a 20-point line, but we can approximate
    predictions for other lines by adjusting the threshold.
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract features in the correct order
    feature_order = [
        'minutes',
        'rebounds',
        'assists',
        'field_goals_attempted',
        'three_pa',
        'free_throws_attempted'
    ]
    
    try:
        feature_vector = [request.features[feat] for feat in feature_order]
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature: {e}"
        )
    
    # Convert to tensor
    features_tensor = torch.FloatTensor([feature_vector])
    
    # Make prediction
    with torch.no_grad():
        prob_over = model(features_tensor).item()
    
    # Adjust threshold based on points_line
    # Simple heuristic: if line is higher than 20, require higher probability
    # if line is lower than 20, require lower probability
    base_line = 20.0
    line_diff = request.points_line - base_line
    threshold = 0.5 + (line_diff / 100.0)  # Adjust threshold by 1% per point difference
    threshold = max(0.3, min(0.7, threshold))  # Clamp between 0.3 and 0.7
    
    # Make prediction
    prediction = "over" if prob_over >= threshold else "under"
    
    return PredictionResponse(
        prob_over=round(prob_over, 4),
        prediction=prediction,
        points_line=request.points_line
    )


@app.get("/example")
async def get_example():
    """Return an example request body for the /predict endpoint."""
    return {
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)

