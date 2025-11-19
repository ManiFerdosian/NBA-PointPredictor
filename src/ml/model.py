"""
PyTorch model definition for NBA over/under prediction.
"""
import torch
import torch.nn as nn


class NBAOverUnderModel(nn.Module):
    """
    Simple feedforward neural network for binary classification.
    Architecture: Linear -> ReLU -> Linear -> Sigmoid
    """
    
    def __init__(self, input_size: int = 6, hidden_size: int = 32):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
        """
        super(NBAOverUnderModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

