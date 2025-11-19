"""
Training script for NBA over/under prediction model.
"""
import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.ml.model import NBAOverUnderModel


def load_data_from_db(db_path: str):
    """
    Load training data from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        X: Feature matrix (numpy array)
        y: Target labels (numpy array)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query player_games table
    query = """
        SELECT minutes, rebounds, assists, field_goals_attempted, 
               three_pa, free_throws_attempted, over_20
        FROM player_games
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to numpy arrays
    data = np.array(rows, dtype=np.float32)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    
    return X, y


def train_model(X_train, y_train, X_val, y_val, model_path: str, epochs: int = 50):
    """
    Train the PyTorch model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_path: Path to save the trained model
        epochs: Number of training epochs
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = NBAOverUnderModel(input_size=input_size, hidden_size=32)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Training model for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
                # Calculate accuracy
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = (val_preds == y_val_tensor).float().mean()
                
                train_preds = (outputs > 0.5).float()
                train_accuracy = (train_preds == y_train_tensor).float().mean()
                
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {loss.item():.4f}, Train Acc: {train_accuracy.item():.4f}")
                print(f"  Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy.item():.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    # Get paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "db.nba.sqlite"
    model_path = project_root / "models" / "nba_over20_model.pt"
    
    # Load data
    print("Loading data from database...")
    X, y = load_data_from_db(str(db_path))
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Train model
    train_model(X_train, y_train, X_val, y_val, str(model_path), epochs=50)

