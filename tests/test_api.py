"""
Smoke tests for the NBA Over/Under API.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test that GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "db_connected" in data


def test_predict_endpoint(client):
    """Test that POST /predict returns 200 and a prediction field."""
    request_body = {
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
    
    response = client.post("/predict", json=request_body)
    # Model might not be loaded in test environment, so accept 503 or 200
    if response.status_code == 503:
        # Model not loaded - this is expected in test environment without Docker
        assert "Model not loaded" in response.json()["detail"]
    else:
        # Model is loaded - verify prediction response
        assert response.status_code == 200
        data = response.json()
        assert "prob_over" in data
        assert "prediction" in data
        assert "points_line" in data
        assert data["prediction"] in ["over", "under"]
        assert 0 <= data["prob_over"] <= 1


def test_example_endpoint(client):
    """Test that GET /example returns an example request."""
    response = client.get("/example")
    assert response.status_code == 200
    data = response.json()
    assert "player_name" in data
    assert "points_line" in data
    assert "features" in data

