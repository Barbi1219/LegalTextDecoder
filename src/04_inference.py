"""
Legal Text Decoder - Inference
==============================
Script for running predictions on new, unseen data WITHOUT labels.

Usage:
  python 04_inference.py                    # Demo mode (samples from test)
  python 04_inference.py --interactive      # Interactive text input
  python 04_inference.py --file input.csv   # Batch inference on CSV
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from models import (
    MostFrequentClassBaseline, TextComplexityClassifier,
    MLPClassifier, NeuralNetworkTrainer
)

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR, RATING_LABELS
from utils import setup_logger, print_separator, extract_text_features, load_model

logger = setup_logger()


def load_best_model(model_name: str = 'gradient_boosting'):
    """Load the best performing model."""
    model_path = MODEL_DIR / f"{model_name}_model.pkl"
    
    if not model_path.exists():
        available_models = list(MODEL_DIR.glob("*_model.pkl"))
        if available_models:
            model_path = available_models[0]
            model_name = model_path.stem.replace('_model', '')
            logger.info(f"Using available model: {model_name}")
        else:
            raise FileNotFoundError("No models found. Run 02_training.py first.")
    
    model = load_model(model_path)
    logger.info(f"Loaded model: {model_name}")
    
    return model, model_name


def predict_single_text(model, text: str) -> Dict[str, Any]:
    """
    Predict rating for a single text (NO LABEL REQUIRED).
    
    Parameters:
    -----------
    model : Any
        Trained model
    text : str
        Input text to classify
    
    Returns:
    --------
    Dict[str, Any]
        Prediction result with rating, label, and confidence
    """
    # Extract features
    features = extract_text_features(text)
    X = np.array([list(features.values())])
    
    # Get prediction
    prediction = int(np.clip(model.predict(X)[0], 1, 5))
    label = RATING_LABELS[prediction - 1]
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
            if probs is not None:
                probabilities = {i+1: float(p) for i, p in enumerate(probs[0])}
        except:
            pass
    
    return {
        'text': text,
        'text_preview': text[:100] + '...' if len(text) > 100 else text,
        'prediction': prediction,
        'label': label,
        'probabilities': probabilities,
        'features': features
    }


def print_prediction_result(result: Dict[str, Any], show_features: bool = True):
    """Print formatted prediction result."""
    logger.info(f"\nText: {result['text_preview']}")
    logger.info(f"Prediction: {result['prediction']} - {result['label']}")
    
    if result['probabilities']:
        logger.info("\nConfidence scores:")
        for rating in range(1, 6):
            prob = result['probabilities'].get(rating, 0.0)
            bar = "█" * int(prob * 30)
            logger.info(f"  Rating {rating}: {prob:.1%} {bar}")
    
    if show_features:
        features = result['features']
        logger.info("\nText statistics:")
        logger.info(f"  Characters: {features['char_count']}")
        logger.info(f"  Words: {features['word_count']}")
        logger.info(f"  Avg word length: {features['avg_word_length']:.2f}")
        logger.info(f"  Complex word ratio: {features['complex_word_ratio']:.1%}")
        logger.info(f"  Legal terms: {features['legal_term_count']}")


def interactive_inference():
    """Interactive mode - enter text manually."""
    print_separator("=")
    logger.info("INTERACTIVE INFERENCE MODE")
    print_separator("=")
    logger.info("\nEnter legal text to predict comprehensibility (1-5 scale).")
    logger.info("Type 'quit' to exit.\n")
    
    model, model_name = load_best_model()
    logger.info(f"Using model: {model_name}\n")
    
    # Example texts
    examples = {
        '1': "A szolgáltató a szerződés teljesítése során a jogszabályok keretei között eljárva jogosult harmadik személy közreműködését igénybe venni.",
        '2': "Ha kérdése van, írjon nekünk emailben."
    }
    
    logger.info("Try these examples (type the number):")
    for key, text in examples.items():
        logger.info(f"  {key}. {text[:60]}...")
    
    while True:
        print_separator("-")
        text = input("\nEnter text (or 'quit'): ").strip()
        
        if text.lower() == 'quit':
            break
        
        # Check if example
        if text in examples:
            text = examples[text]
            logger.info(f"\nUsing example: {text[:80]}...")
        
        if not text:
            logger.info("Please enter some text.")
            continue
        
        # Predict (NO LABEL NEEDED!)
        result = predict_single_text(model, text)
        print_prediction_result(result)
    
    logger.info("\nInference session ended.")


def batch_inference_from_csv(input_csv: Path, output_csv: Path = None):
    """
    Run inference on CSV file (NO 'rating' column required).
    
    Parameters:
    -----------
    input_csv : Path
        CSV file with 'text' column
    output_csv : Path, optional
        Output path (default: input_predictions.csv)
    """
    print_separator("=")
    logger.info("BATCH INFERENCE FROM CSV")
    print_separator("=")
    
    # Load data
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    # Check for text column
    if 'text' not in df.columns:
        raise ValueError("CSV must contain 'text' column")
    
    logger.info(f"Loaded {len(df)} texts from {input_csv.name}")
    
    # Load model
    model, model_name = load_best_model()
    
    # Run predictions
    logger.info("Running predictions...")
    results = []
    
    for idx, text in enumerate(df['text']):
        if pd.isna(text) or not text:
            continue
        
        result = predict_single_text(model, text)
        results.append({
            'text': text,
            'prediction': result['prediction'],
            'label': result['label']
        })
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} texts")
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save results
    if output_csv is None:
        output_csv = OUTPUT_DIR / f"{input_csv.stem}_predictions.csv"
    
    output_df.to_csv(output_csv, index=False)
    logger.info(f"\nPredictions saved to: {output_csv}")
    
    # Summary statistics
    print_separator("-")
    logger.info("PREDICTION SUMMARY")
    print_separator("-")
    
    logger.info(f"\nTotal predictions: {len(output_df)}")
    logger.info("\nPrediction distribution:")
    pred_counts = output_df['prediction'].value_counts().sort_index()
    for rating in range(1, 6):
        count = pred_counts.get(rating, 0)
        pct = (count / len(output_df)) * 100 if len(output_df) > 0 else 0
        logger.info(f"  Rating {rating}: {count:4d} ({pct:5.1f}%)")
    
    return output_df


def demo_inference():
    """
    Demo mode: sample predictions from test set for demonstration.
    (This uses labeled data ONLY for comparison - not required for inference!)
    """
    print_separator("=")
    logger.info("INFERENCE DEMO")
    logger.info("(Using test data samples to demonstrate inference capability)")
    print_separator("=")
    
    # Load model
    model, model_name = load_best_model()
    
    # Load test data (for demo only!)
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    
    if not test_path.exists():
        logger.warning("Test data not found. Run preprocessing first.")
        return None
    
    test_df = pd.read_csv(test_path)
    
    # Sample 5 texts
    sample_df = test_df.sample(n=min(5, len(test_df)), random_state=42)
    
    logger.info(f"Running inference on {len(sample_df)} sample texts:\n")
    
    correct = 0
    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        print_separator("-")
        logger.info(f"Sample {idx}:")
        
        text = row['text']
        result = predict_single_text(model, text)
        
        logger.info(f"Text: {result['text_preview']}")
        logger.info(f"Predicted: {result['prediction']} - {result['label']}")
        
        # Compare with true rating (demo only!)
        if 'rating' in row:
            true_rating = int(row['rating'])
            true_label = RATING_LABELS[true_rating - 1]
            logger.info(f"True:      {true_rating} - {true_label}")
            
            if result['prediction'] == true_rating:
                logger.info("✓ Correct prediction!")
                correct += 1
            else:
                diff = result['prediction'] - true_rating
                logger.info(f"✗ Error: {diff:+d}")
    
    print_separator("=")
    logger.info(f"Demo accuracy: {correct}/{len(sample_df)} = {correct/len(sample_df):.1%}")
    print_separator("=")


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description='Legal Text Decoder - Inference')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode (enter text manually)')
    parser.add_argument('--file', type=str,
                       help='Input CSV file with "text" column (no labels needed)')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for predictions')
    parser.add_argument('--model', type=str, default='gradient_boosting',
                       help='Model to use (default: gradient_boosting)')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        interactive_inference()
    
    elif args.file:
        # Batch inference from file
        input_path = Path(args.file)
        output_path = Path(args.output) if args.output else None
        batch_inference_from_csv(input_path, output_path)
    
    else:
        # Demo mode (default)
        demo_inference()
    
    print_separator("=")
    logger.info("INFERENCE COMPLETED")
    print_separator("=")


if __name__ == "__main__":
    main()