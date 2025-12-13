"""
Legal Text Decoder - Model Evaluation
=====================================
Script for evaluating trained models with comprehensive metrics.

This script implements advanced evaluation including:
1. Standard metrics: MSE, MAE, RMSE, Accuracy
2. Ordinal-specific metrics: Quadratic Weighted Kappa, Ordinal MSE
3. Classification metrics: Confusion Matrix, Classification Report
4. Comparative analysis across all models
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score
)

from config import (
    MODEL_DIR, OUTPUT_DIR,
    RATING_MIN, RATING_MAX
)
from utils import (
    setup_logger, print_separator, format_metrics
)

# Setup logger
logger = setup_logger()


# =============================================================================
# ORDINAL METRICS
# =============================================================================

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Quadratic Weighted Kappa (QWK).
    
    QWK is a metric specifically designed for ordinal classification.
    It penalizes predictions that are further from the true label more heavily.
    
    Range: [-1, 1], where:
    - 1 = perfect agreement
    - 0 = agreement equivalent to chance
    - Negative = less than chance agreement
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    
    Returns:
    --------
    float
        QWK score
    """
    min_rating = min(RATING_MIN, min(y_true), min(y_pred))
    max_rating = max(RATING_MAX, max(y_true), max(y_pred))
    
    num_ratings = max_rating - min_rating + 1
    
    # Build the weight matrix
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    
    # Build the observed histogram matrix
    y_true_shifted = y_true - min_rating
    y_pred_shifted = y_pred - min_rating
    
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)
    conf_mat = np.zeros((num_ratings, num_ratings))
    
    for t, p in zip(y_true_shifted, y_pred_shifted):
        hist_true[int(t)] += 1
        hist_pred[int(p)] += 1
        conf_mat[int(t), int(p)] += 1
    
    # Normalize
    conf_mat = conf_mat / conf_mat.sum()
    hist_true = hist_true / hist_true.sum()
    hist_pred = hist_pred / hist_pred.sum()
    
    # Build expected histogram matrix
    expected = np.outer(hist_true, hist_pred)
    
    # Calculate kappa
    numerator = np.sum(weights * conf_mat)
    denominator = np.sum(weights * expected)
    
    if denominator == 0:
        return 0.0
    
    return 1 - (numerator / denominator)


def ordinal_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate ordinal-aware Mean Squared Error.
    
    This is the same as regular MSE but explicitly interprets
    predictions as ordinal values on the rating scale.
    """
    return mean_squared_error(y_true, y_pred)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Expressed as a percentage of the rating scale.
    """
    scale_range = RATING_MAX - RATING_MIN
    mae = mean_absolute_error(y_true, y_pred)
    return (mae / scale_range) * 100


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 1) -> float:
    """
    Calculate accuracy allowing for off-by-one predictions.
    
    This is useful for ordinal data where predicting 3 when true is 4
    is better than predicting 1 when true is 4.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    tolerance : int
        Maximum allowed difference for correct prediction
    
    Returns:
    --------
    float
        Adjacent accuracy score
    """
    correct = np.abs(y_true - y_pred) <= tolerance
    return np.mean(correct)


def macro_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate class-balanced (macro) MAE.
    
    This gives equal weight to each class regardless of frequency.
    """
    classes = np.unique(y_true)
    class_maes = []
    
    for cls in classes:
        mask = y_true == cls
        if mask.sum() > 0:
            class_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            class_maes.append(class_mae)
    
    return np.mean(class_maes)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Returns dictionary with:
    - Regression metrics: MSE, RMSE, MAE
    - Classification metrics: Accuracy, F1
    - Ordinal metrics: QWK, Adjacent Accuracy
    """
    metrics = {}
    
    # Regression metrics
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    metrics['Macro_MAE'] = macro_mae(y_true, y_pred)
    
    # Classification metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1_Macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['F1_Weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Ordinal-specific metrics
    metrics['QWK'] = quadratic_weighted_kappa(y_true, y_pred)
    metrics['Adjacent_Accuracy'] = adjacent_accuracy(y_true, y_pred, tolerance=1)
    
    return metrics


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Print formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    
    logger.info(f"\nConfusion Matrix - {model_name}:")
    logger.info("-" * 50)
    
    # Header
    header = "True\\Pred |  1  |  2  |  3  |  4  |  5  | Total"
    logger.info(header)
    logger.info("-" * 50)
    
    # Rows
    for i, row in enumerate(cm):
        row_str = f"    {i+1}     | {row[0]:3d} | {row[1]:3d} | {row[2]:3d} | {row[3]:3d} | {row[4]:3d} | {row.sum():5d}"
        logger.info(row_str)
    
    logger.info("-" * 50)
    col_totals = cm.sum(axis=0)
    total_str = f"  Total   | {col_totals[0]:3d} | {col_totals[1]:3d} | {col_totals[2]:3d} | {col_totals[3]:3d} | {col_totals[4]:3d} | {cm.sum():5d}"
    logger.info(total_str)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Print classification report."""
    logger.info(f"\nClassification Report - {model_name}:")
    logger.info("-" * 60)
    
    report = classification_report(
        y_true, y_pred,
        labels=[1, 2, 3, 4, 5],
        target_names=['Rating 1', 'Rating 2', 'Rating 3', 'Rating 4', 'Rating 5'],
        zero_division=0
    )
    logger.info(report)


def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Analyze prediction errors."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    logger.info(f"\nError Analysis - {model_name}:")
    logger.info("-" * 50)
    # Error distribution with direction
    logger.info(f"Signed Error Distribution:")
    logger.info("  (Negative = Underestimate, Positive = Overestimate)")
    logger.info("")
    
    logger.info(f"Error distribution:")
    for err in range(-4, 5):
        count = np.sum(errors == err)
        if count > 0:
            pct = (count / len(errors)) * 100
            bar = "█" * int(pct / 2)
            logger.info(f"  Error {err:+2d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    logger.info(f"\nAbsolute error distribution:")
    for err in range(5):
        count = np.sum(abs_errors == err)
        pct = (count / len(errors)) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  |Error| = {err}: {count:4d} ({pct:5.1f}%) {bar}")


def evaluate_all_models(predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict]:
    """
    Evaluate all models and compare results.
    
    Returns dictionary with metrics for each model.
    """
    all_results = {}
    
    for model_name, y_pred in predictions.items():
        metrics = calculate_all_metrics(y_true, y_pred)
        all_results[model_name] = metrics
    
    return all_results


def print_comparison_table(results: Dict[str, Dict]):
    """Print comparison table of all models."""
    print_separator("=")
    logger.info("MODEL COMPARISON TABLE")
    print_separator("=")
    
    # Get metric names
    first_model = list(results.keys())[0]
    metric_names = list(results[first_model].keys())
    
    # Create DataFrame for comparison
    comparison_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print table
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Find best model for each metric
    logger.info("\n\nBest Model per Metric:")
    logger.info("-" * 50)
    
    # For these metrics, lower is better
    lower_is_better = ['MSE', 'RMSE', 'MAE', 'MAPE', 'Macro_MAE']
    
    for metric in metric_names:
        values = {model: results[model][metric] for model in results}
        if metric in lower_is_better:
            best_model = min(values, key=values.get)
            best_value = values[best_model]
        else:
            best_model = max(values, key=values.get)
            best_value = values[best_model]
        
        logger.info(f"  {metric}: {best_model} ({best_value:.4f})")
    
    return comparison_df


def main():
    """Main evaluation pipeline."""
    print_separator("=")
    logger.info("LEGAL TEXT DECODER - MODEL EVALUATION")
    print_separator("=")
    
    # Load predictions
    predictions_path = OUTPUT_DIR / "predictions.csv"
    
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Predictions not found. Run 02_training.py first."
        )
    
    predictions_df = pd.read_csv(predictions_path)
    y_true = predictions_df['y_true'].values
    
    logger.info(f"\nLoaded predictions for {len(y_true)} test samples")
    
    # Get model predictions
    model_names = [col for col in predictions_df.columns if col != 'y_true']
    predictions = {name: predictions_df[name].values for name in model_names}
    
    logger.info(f"Models to evaluate: {model_names}")
    
    # =========================================================================
    # Evaluate Each Model
    # =========================================================================
    all_results = {}
    
    for model_name in model_names:
        print_separator("=")
        logger.info(f"\nEVALUATION: {model_name.upper()}")
        print_separator("=")
        
        y_pred = predictions[model_name]
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_true, y_pred)
        all_results[model_name] = metrics
        
        # Print metrics
        logger.info("\nMetrics:")
        logger.info("-" * 50)
        logger.info(format_metrics(metrics))
        
        # Confusion matrix
        print_confusion_matrix(y_true, y_pred, model_name)
        
        # Classification report
        print_classification_report(y_true, y_pred, model_name)
        
        # Error analysis
        analyze_errors(y_true, y_pred, model_name)
    
    # =========================================================================
    # Model Comparison
    # =========================================================================
    comparison_df = print_comparison_table(all_results)
    
    # Save comparison results
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\n Comparison saved to {comparison_path}")
    
    # =========================================================================
    # Evaluation Criteria Definition
    # =========================================================================
    print_separator("=")
    logger.info("\nEVALUATION CRITERIA DEFINITION")
    print_separator("=")
    
    logger.info("""
The following metrics are used to evaluate model performance:

1. REGRESSION METRICS (treat ratings as continuous):
   - MSE: Mean Squared Error - penalizes large errors more heavily
   - RMSE: Root Mean Squared Error - same scale as ratings
   - MAE: Mean Absolute Error - average error magnitude
   - MAPE: Mean Absolute Percentage Error (relative to scale)
   - Macro MAE: Class-balanced MAE (equal weight per class)

2. CLASSIFICATION METRICS (treat ratings as discrete classes):
   - Accuracy: Exact match rate
   - F1 Macro: Harmonic mean of precision/recall (class-balanced)
   - F1 Weighted: F1 weighted by class frequency

3. ORDINAL-SPECIFIC METRICS (account for rating order):
   - QWK: Quadratic Weighted Kappa - penalizes distant errors more
         Range [-1, 1], higher is better, 1 = perfect agreement
   - Adjacent Accuracy: Allows off-by-one predictions as correct
         Useful since 3 vs 4 is less severe than 1 vs 5

PRIMARY EVALUATION CRITERION: 
   QWK (Quadratic Weighted Kappa) is recommended as the primary metric
   because it properly handles the ordinal nature of the rating scale
   and penalizes larger prediction errors more heavily.

SECONDARY METRICS:
   - MAE for interpretability (average error in rating units)
   - Accuracy for exact match performance
   - Adjacent Accuracy for practical usefulness
""")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print_separator("=")
    logger.info("EVALUATION SUMMARY")
    print_separator("=")
    
    # Find overall best model (using QWK as primary metric)
    best_qwk_model = max(all_results, key=lambda x: all_results[x]['QWK'])
    best_accuracy_model = max(all_results, key=lambda x: all_results[x]['Accuracy'])
    best_mae_model = min(all_results, key=lambda x: all_results[x]['MAE'])
    
    logger.info(f"\nBest model by QWK: {best_qwk_model} ({all_results[best_qwk_model]['QWK']:.4f})")
    logger.info(f"Best model by Accuracy: {best_accuracy_model} ({all_results[best_accuracy_model]['Accuracy']:.4f})")
    logger.info(f"Best model by Adjacent Accuracy: {best_accuracy_model} ({all_results[best_accuracy_model]['Adjacent_Accuracy']:.4f})")
    logger.info(f"Best model by MAE: {best_mae_model} ({all_results[best_mae_model]['MAE']:.4f})")
    
    
    # Improvement over baseline
    baseline_qwk = all_results['baseline']['QWK']
    baseline_mae = all_results['baseline']['MAE']
    
    logger.info(f"\nImprovement over baseline:")
    for model in model_names:
        if model != 'baseline':
            qwk_improvement = all_results[model]['QWK'] - baseline_qwk
            mae_improvement = baseline_mae - all_results[model]['MAE']
            logger.info(f"  {model}:")
            logger.info(f"    QWK: {'+' if qwk_improvement >= 0 else ''}{qwk_improvement:.4f}")
            logger.info(f"    MAE: {'+' if mae_improvement >= 0 else ''}{mae_improvement:.4f}")
    
    print_separator("=")
    logger.info("EVALUATION COMPLETED")
    print_separator("=")
    
    return all_results


if __name__ == "__main__":
    results = main()
