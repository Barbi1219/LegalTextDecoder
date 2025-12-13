"""
Legal Text Decoder - Model Classes
==================================
Reusable model class definitions for training and inference.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    RANDOM_SEED, HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE
)

from utils import (
    setup_logger
)

# Setup logger
logger = setup_logger()

class MostFrequentClassBaseline:
    """
    Baseline Model: Always predicts the most frequent class.
    
    This is the simplest possible model and serves as the lower bound
    for model performance.
    """
    
    def __init__(self):
        self.most_frequent = None
        self.class_counts = None
    
    def fit(self, X, y):
        """Train by finding the most frequent class."""
        self.class_counts = Counter(y)
        self.most_frequent = self.class_counts.most_common(1)[0][0]
        return self
    
    def predict(self, X):
        """Predict the most frequent class for all samples."""
        return np.full(len(X), self.most_frequent)
    
    def __repr__(self):
        return f"MostFrequentClassBaseline(prediction={self.most_frequent})"


class TextComplexityClassifier:
    """Wrapper for sklearn classifiers with standardization."""
    
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED
            )
        
        elif model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=50,          # Reduced from 100
                max_depth=5,              # Reduced from 10
                min_samples_split=10,     # At least 10 samples to split
                min_samples_leaf=5,       # At least 5 samples in leaf
                max_features='sqrt',      # Use sqrt(n_features) per tree
                random_state=RANDOM_SEED,
            )

        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=40,          # Reduced from 100
                max_depth=3,              # Reduced from 5
                learning_rate=0.05,       # Reduced from 0.1 (slower learning)
                min_samples_split=10,  
                min_samples_leaf=5,      
                subsample=0.8,            # Use 80% of data per tree
                max_features='sqrt',      
                random_state=RANDOM_SEED
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """Fit the model with feature scaling."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Predict with feature scaling."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities with feature scaling."""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return None


class MLPClassifier(nn.Module):
    """
    Neural Network Model: Multi-Layer Perceptron for rating prediction.
    
    Architecture:
    - Input layer: feature_dim
    - Hidden layer 1: hidden_dim with ReLU and Dropout
    - Hidden layer 2: hidden_dim // 2 with ReLU and Dropout
    - Output layer: num_classes (for classification)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, 
                 num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
        super(MLPClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkTrainer:
    """
    Wrapper class for training PyTorch neural network.
    """
    
    def __init__(self, input_dim: int, epochs: int = EPOCHS, 
                 batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = MLPClassifier(input_dim)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_losses = []
        self.val_losses = []
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the neural network."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors (y is 1-5, convert to 0-4 for CrossEntropyLoss)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y - 1).to(self.device)  # 1-5 -> 0-4
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.LongTensor(y_val - 1).to(self.device)
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            
            # Validation loss
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    self.val_losses.append(val_loss)
                self.model.train()
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_info = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}{val_info}")
        
        return self
    
    def predict(self, X):
        """Predict ratings."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        
        # Convert back from 0-4 to 1-5
        return predictions.cpu().numpy() + 1