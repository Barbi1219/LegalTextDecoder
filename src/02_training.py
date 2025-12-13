"""
Legal Text Decoder - Model Training
===================================
Script for training models to predict legal text comprehensibility.

This script implements incremental model development:
1. Baseline Model: Most Frequent Class
2. Feature-based Model: Text Complexity Features + Logistic Regression
3. Neural Network Model: MLP with PyTorch
4. Advanced Model: Ensemble approach

Each model builds upon the previous, following incremental development principles.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED,
    HIDDEN_DIM, DROPOUT_RATE, NUM_CLASSES, FEATURE_NAMES
)
from utils import (
    setup_logger, print_separator,
    save_model, load_model
)

from models import (
    MostFrequentClassBaseline, TextComplexityClassifier,
    MLPClassifier, NeuralNetworkTrainer
)

# Setup logger
logger = setup_logger()

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data."""
    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run 01_data_preprocessing.py first."
        )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature column names from DataFrame."""
    # Use predefined feature names that exist in the DataFrame
    feature_cols = [col for col in FEATURE_NAMES if col in df.columns]
    
    # Add any additional numeric columns that might be features
    additional_features = ['text_length', 'word_count']
    for col in additional_features:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
    
    return feature_cols


def train_all_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train all models incrementally.
    
    Returns dictionary with all trained models and their predictions.
    """
    
    # Get features and labels
    feature_cols = get_feature_columns(train_df)
    logger.info(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['rating'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['rating'].values
    
    models = {}
    predictions = {}
    
    # =========================================================================
    # Model 1: Baseline - Most Frequent Class
    # =========================================================================
    print_separator("-")
    logger.info("\n[MODEL 1] BASELINE: Most Frequent Class")
    print_separator("-")
    
    baseline = MostFrequentClassBaseline()
    baseline.fit(X_train, y_train)
    
    logger.info(f"Most frequent class in training data: {baseline.most_frequent}")
    logger.info(f"Class distribution: {dict(baseline.class_counts)}")
    
    predictions['baseline'] = baseline.predict(X_test)
    models['baseline'] = baseline
    logger.info("Baseline model trained")
    
    # Baseline accuracy (on training data)
    baseline_train_acc = (y_train == baseline.most_frequent).mean()
    logger.info(f"Baseline training accuracy: {baseline_train_acc:.4f}")
    logger.info(f"  (Always predicts class {baseline.most_frequent} - represents {baseline_train_acc*100:.1f}% of training data)")
    
    
    # =========================================================================
    # Model 2: Logistic Regression with Text Features
    # =========================================================================
    print_separator("-")
    logger.info("\n[MODEL 2] LOGISTIC REGRESSION with Text Complexity Features")
    print_separator("-")
    
    logreg = TextComplexityClassifier(model_type='logistic')
    logreg.fit(X_train, y_train)
    
    predictions['logistic'] = logreg.predict(X_test)
    models['logistic'] = logreg
    logger.info("Logistic Regression model trained")
    
    # Training accuracy
    train_acc = (logreg.predict(X_train) == y_train).mean()
    logger.info(f"Training accuracy: {train_acc:.4f}")
    
    # Cross-validation score
    cv_scores_logreg = cross_val_score(
        logreg.model,  
        logreg.scaler.transform(X_train),  
        y_train,
        cv=min(5, min(Counter(y_train).values())),
        scoring='accuracy'
    )
    logger.info(f"Cross-validation accuracy: {cv_scores_logreg.mean():.4f} (+/- {cv_scores_logreg.std()*2:.4f})")
    
    # =========================================================================
    # Model 3: Random Forest
    # =========================================================================
    print_separator("-")
    logger.info("\n[MODEL 3] RANDOM FOREST Classifier")
    print_separator("-")
    
    rf = TextComplexityClassifier(model_type='rf')
    
    # Log parameters
    logger.info("Configuration:")
    logger.info(f"  n_estimators: {rf.model.n_estimators}")      
    logger.info(f"  max_depth: {rf.model.max_depth}")           
    logger.info(f"  min_samples_split: {rf.model.min_samples_split}")
    logger.info(f"  min_samples_leaf: {rf.model.min_samples_leaf}")
    
    

    rf.fit(X_train, y_train)
    
    predictions['random_forest'] = rf.predict(X_test)
    models['random_forest'] = rf
    logger.info("Random Forest model trained")
    
    train_acc_rf = (rf.predict(X_train) == y_train).mean()
    logger.info(f"\nTraining accuracy: {train_acc_rf:.4f}")
    
        # Cross-validation
    cv_scores_rf = cross_val_score(
        rf.model,  
        rf.scaler.transform(X_train),  
        y_train,
        cv=min(5, min(Counter(y_train).values())),
        scoring='accuracy'
    )
    logger.info(f"Cross-validation accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std()*2:.4f})")
    
    # Feature importance
    if hasattr(rf.model, 'feature_importances_'):
        importances = rf.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        logger.info("\nTop 5 important features:")
        for i in sorted_idx[:5]:
            logger.info(f"  {feature_cols[i]}: {importances[i]:.4f}")
    
    
    # =========================================================================
    # Model 4: Gradient Boosting
    # =========================================================================
    print_separator("-")
    logger.info("\n[MODEL 4] GRADIENT BOOSTING Classifier")
    print_separator("-")
    
    gb = TextComplexityClassifier(model_type='gb')
    
    logger.info("Configuration:")
    logger.info(f"  n_estimators: {gb.model.n_estimators}")      # ← VALÓDI érték
    logger.info(f"  max_depth: {gb.model.max_depth}")            # ← VALÓDI érték
    logger.info(f"  min_samples_split: {gb.model.min_samples_split}")
    logger.info(f"  min_samples_leaf: {gb.model.min_samples_leaf}")
    
    gb.fit(X_train, y_train)
    
    predictions['gradient_boosting'] = gb.predict(X_test)
    models['gradient_boosting'] = gb
    logger.info("Gradient Boosting model trained")
    
    # Training accuracy
    train_acc_gb = (gb.predict(X_train) == y_train).mean()
    logger.info(f"\nTraining accuracy: {train_acc_gb:.4f}")
    
        # Cross-validation
    cv_scores_gb = cross_val_score(
        gb.model,  
        gb.scaler.transform(X_train), 
        y_train,
        cv=min(5, min(Counter(y_train).values())),
        scoring='accuracy'
    )
    logger.info(f"Cross-validation accuracy: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std()*2:.4f})")
    
        # Feature importance
    if hasattr(gb.model, 'feature_importances_'):
        importances_gb = gb.model.feature_importances_
        sorted_idx_gb = np.argsort(importances_gb)[::-1]
        logger.info("\nTop 5 important features:")
        for i in sorted_idx_gb[:5]:
            logger.info(f"  {feature_cols[i]}: {importances_gb[i]:.4f}")
  
    
    # =========================================================================
    # Model 5: Neural Network (MLP)
    # =========================================================================
    print_separator("-")
    logger.info("\n[MODEL 5] NEURAL NETWORK (MLP)")
    print_separator("-")
    
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Hidden dimension: {HIDDEN_DIM}")
    logger.info(f"  Dropout: {DROPOUT_RATE}")
    logger.info(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Split some training data for validation
    val_size = int(len(X_train) * 0.2)
    indices = np.random.permutation(len(X_train))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train_nn = X_train[train_indices]
    y_train_nn = y_train[train_indices]
    X_val_nn = X_train[val_indices]
    y_val_nn = y_train[val_indices]
    
    nn_trainer = NeuralNetworkTrainer(
        input_dim=len(feature_cols),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    
    logger.info("\nTraining neural network...")
    nn_trainer.fit(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
    
    predictions['neural_network'] = nn_trainer.predict(X_test)
    models['neural_network'] = nn_trainer
    logger.info("Neural Network model trained")
    
    # Print model architecture
    logger.info(f"\nModel architecture:")
    logger.info(str(nn_trainer.model))
    
    # Count parameters
    total_params = sum(p.numel() for p in nn_trainer.model.parameters())
    trainable_params = sum(p.numel() for p in nn_trainer.model.parameters() if p.requires_grad)
    logger.info(f"\nTotal parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    
    # =========================================================================
    # Save Models
    # =========================================================================
    print_separator("-")
    logger.info("\n[SAVING MODELS]")
    print_separator("-")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_path = MODEL_DIR / f"{name}_model.pkl"
        save_model(model, model_path)
    
    # Save predictions for evaluation
    predictions_df = pd.DataFrame(predictions)
    predictions_df['y_true'] = y_test
    predictions_path = OUTPUT_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f" Predictions saved to {predictions_path}")
    
    print_separator("=")
    logger.info("MODEL TRAINING COMPLETED")
    
    return {
        'models': models,
        'predictions': predictions,
        'y_test': y_test,
        'feature_cols': feature_cols
    }


def main():
    """Main training pipeline."""
    print_separator("=")
    logger.info("MODEL TRAINING - INCREMENTAL DEVELOPMENT")
    print_separator("=")
    
    # Load data
    logger.info("\n[LOADING DATA]")
    train_df, test_df = load_processed_data()
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Train all models
    results = train_all_models(train_df, test_df)
    
    return results


if __name__ == "__main__":
    results = main()
