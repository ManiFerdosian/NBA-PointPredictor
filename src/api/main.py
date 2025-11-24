"""
FastAPI application for NBA over/under prediction.
"""
import os
import sqlite3
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict

from src.ml.model import NBAOverUnderModel


# Environment variables
DB_PATH = os.getenv("DB_PATH", "data/db.nba.sqlite")
MODEL_PATH = os.getenv("MODEL_PATH", "models/nba_over30_model.pt")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Initialize FastAPI app
app = FastAPI(title="NBA Over/Under Predictor API")

# Setup templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Global model variable
model = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_db_file() -> Path:
    """Get the database file path."""
    return get_project_root() / DB_PATH


def get_player_feature_averages(player_name: str, n_games: int = 10) -> Optional[Dict[str, float]]:
    """
    Get average feature values for a player's last N games.
    
    Args:
        player_name: Name of the player
        n_games: Number of recent games to average (default: 10)
    
    Returns:
        Dictionary with averaged features, or None if no data found
    """
    db_file = get_db_file()
    if not db_file.exists():
        return None
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Query to get averages from last N games
        # First get the last N games, then compute averages
        query = """
            SELECT
                AVG(minutes) as avg_minutes,
                AVG(rebounds) as avg_rebounds,
                AVG(assists) as avg_assists,
                AVG(field_goals_attempted) as avg_field_goals_attempted,
                AVG(three_pa) as avg_three_pa,
                AVG(free_throws_attempted) as avg_free_throws_attempted
            FROM (
                SELECT minutes, rebounds, assists, field_goals_attempted, three_pa, free_throws_attempted
                FROM player_games
                WHERE player_name = ?
                ORDER BY game_date DESC
                LIMIT ?
            )
        """
        
        cursor.execute(query, (player_name, n_games))
        row = cursor.fetchone()
        conn.close()
        
        if row is None or row[0] is None:
            return None
        
        return {
            'minutes': float(row[0]) if row[0] is not None else 0.0,
            'rebounds': float(row[1]) if row[1] is not None else 0.0,
            'assists': float(row[2]) if row[2] is not None else 0.0,
            'field_goals_attempted': float(row[3]) if row[3] is not None else 0.0,
            'three_pa': float(row[4]) if row[4] is not None else 0.0,
            'free_throws_attempted': float(row[5]) if row[5] is not None else 0.0
        }
    except Exception as e:
        print(f"Error querying database: {e}")
        return None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    player_name: str


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prob_over: float
    prediction: str
    player_name: str
    averaged_features: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model
    
    # Compute project root
    project_root = get_project_root()
    
    # Load model
    model_file = project_root / MODEL_PATH
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    print(f"Loading model from {model_file}...")
    model = NBAOverUnderModel(input_size=6, hidden_size=32)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")
    
    # Check database file
    db_file = get_db_file()
    if db_file.exists():
        print(f"Database found at {db_file}")
    else:
        print(f"Warning: Database not found at {db_file}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    
    db_file = get_db_file()
    db_exists = db_file.exists()
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "db_connected": db_exists
    }


@app.get("/players")
async def list_players():
    """List all distinct player names from the database in alphabetical order."""
    db_file = get_db_file()
    if not db_file.exists():
        return {"players": []}
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Get all distinct players, ordered alphabetically (case-insensitive)
        query = """
            SELECT DISTINCT player_name 
            FROM player_games 
            WHERE player_name IS NOT NULL 
            ORDER BY player_name COLLATE NOCASE
        """
        cursor.execute(query)
        players = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return {"players": players}
    except Exception as e:
        print(f"Error querying players: {e}")
        return {"players": []}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict whether a player will score over 30 points in their next game."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get player feature averages from database
    feature_averages = get_player_feature_averages(request.player_name, n_games=10)
    if feature_averages is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for player: {request.player_name}"
        )
    
    # Build feature vector in fixed order
    feature_order = [
        'minutes',
        'rebounds',
        'assists',
        'field_goals_attempted',
        'three_pa',
        'free_throws_attempted'
    ]
    
    feature_vector = [feature_averages[feat] for feat in feature_order]
    
    # Convert to tensor
    features_tensor = torch.FloatTensor([feature_vector])
    
    # Run model prediction to get probability of scoring over 30 points
    with torch.no_grad():
        base_prob = model(features_tensor).item()
    
    # Make prediction using fixed 0.5 threshold
    prediction = "over" if base_prob >= 0.5 else "under"
    
    return PredictionResponse(
        prob_over=round(base_prob, 4),
        prediction=prediction,
        player_name=request.player_name,
        averaged_features=feature_averages
    )


@app.get("/example")
async def get_example():
    """Return an example request body for the /predict endpoint."""
    return {
        "player_name": "LeBron James"
    }


@app.get("/evaluate")
async def evaluate_dataset():
    """Return dataset evaluation statistics."""
    db_file = get_db_file()
    if not db_file.exists():
        return {"error": "Database not found"}
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Get total games
        cursor.execute("SELECT COUNT(*) FROM player_games")
        total_games = cursor.fetchone()[0]
        
        # Get games over/under 30 (stored in over_20 column)
        cursor.execute("SELECT COUNT(*) FROM player_games WHERE over_20 = 1")
        over_30_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_games WHERE over_20 = 0")
        under_30_count = cursor.fetchone()[0]
        
        # Get unique players
        cursor.execute("SELECT COUNT(DISTINCT player_name) FROM player_games")
        unique_players = cursor.fetchone()[0]
        
        # Get average points
        cursor.execute("SELECT AVG(points) FROM player_games")
        avg_points = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_games": total_games,
            "games_over_30": over_30_count,
            "games_under_30": under_30_count,
            "percentage_over_30": round(over_30_count / total_games * 100, 2) if total_games > 0 else 0,
            "percentage_under_30": round(under_30_count / total_games * 100, 2) if total_games > 0 else 0,
            "unique_players": unique_players,
            "average_points": round(avg_points, 2) if avg_points else 0,
            "target_line": 30
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
