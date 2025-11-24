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
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

from src.ml.model import NBAOverUnderModel

# Target line for prediction
TARGET_LINE = 30


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


def train_model(X_train, y_train, X_val, y_val, model_path: str, epochs: int = 100):
    """
    Train the PyTorch model with class weights to handle imbalance.
    
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
    
    # Calculate class weights for imbalanced dataset
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = {int(cls): weight for cls, weight in zip(classes, class_weights)}
    
    # Create weight tensor
    weight_tensor = torch.FloatTensor([class_weights_dict[0], class_weights_dict[1]])
    
    print(f"Class distribution - Under {TARGET_LINE}: {(y_train == 0).sum()}, Over {TARGET_LINE}: {(y_train == 1).sum()}")
    print(f"Class weights - Under {TARGET_LINE}: {class_weights_dict[0]:.4f}, Over {TARGET_LINE}: {class_weights_dict[1]:.4f}")
    
    # Initialize model
    input_size = X_train.shape[1]
    model = NBAOverUnderModel(input_size=input_size, hidden_size=32)
    
    # Loss with class weights (using weighted BCELoss)
    pos_weight = class_weights_dict[1] / class_weights_dict[0]
    
    def weighted_bce_loss(output, target):
        # Manual weighted BCE loss
        loss = -(pos_weight * target * torch.log(output + 1e-8) + 
                 (1 - target) * torch.log(1 - output + 1e-8))
        return loss.mean()
    
    criterion = weighted_bce_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\nTraining model for {epochs} epochs...")
    best_val_acc = 0.0
    
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
                
                # Calculate accuracy (outputs already have sigmoid applied)
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = (val_preds == y_val_tensor).float().mean()
                
                train_preds = (outputs > 0.5).float()
                train_accuracy = (train_preds == y_train_tensor).float().mean()
                
                # Calculate per-class accuracy
                val_over_pred = (val_preds == 1).sum().item()
                val_over_true = (y_val_tensor == 1).sum().item()
                val_under_pred = (val_preds == 0).sum().item()
                val_under_true = (y_val_tensor == 0).sum().item()
                
                val_over_correct = ((val_preds == 1) & (y_val_tensor == 1)).sum().item()
                val_under_correct = ((val_preds == 0) & (y_val_tensor == 0)).sum().item()
                
                val_over_acc = val_over_correct / val_over_true if val_over_true > 0 else 0.0
                val_under_acc = val_under_correct / val_under_true if val_under_true > 0 else 0.0
                
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {loss.item():.4f}, Train Acc: {train_accuracy.item():.4f}")
                print(f"  Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy.item():.4f}")
                print(f"  Val Over Acc: {val_over_acc:.4f} ({val_over_correct}/{val_over_true})")
                print(f"  Val Under Acc: {val_under_acc:.4f} ({val_under_correct}/{val_under_true})")
                
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    print(f"Training model to predict probability of scoring over {TARGET_LINE} points.")
    
    # Get paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "db.nba.sqlite"
    model_path = project_root / "models" / "nba_over30_model.pt"
    
    # Load data
    print("Loading data from database...")
    X, y = load_data_from_db(str(db_path))
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Print dataset statistics
    over_30_count = (y == 1).sum()
    under_30_count = (y == 0).sum()
    total = len(y)
    
    print(f"\nDataset Statistics (Target Line: {TARGET_LINE} points):")
    print(f"Total games: {total}")
    print(f"Games over {TARGET_LINE} points: {over_30_count} ({over_30_count/total*100:.1f}%)")
    print(f"Games under {TARGET_LINE} points: {under_30_count} ({under_30_count/total*100:.1f}%)")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"  Over {TARGET_LINE}: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"  Under {TARGET_LINE}: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples")
    print(f"  Over {TARGET_LINE}: {(y_val == 1).sum()} ({(y_val == 1).sum()/len(y_val)*100:.1f}%)")
    print(f"  Under {TARGET_LINE}: {(y_val == 0).sum()} ({(y_val == 0).sum()/len(y_val)*100:.1f}%)")
    
    # Train model
    train_model(X_train, y_train, X_val, y_val, str(model_path), epochs=100)
