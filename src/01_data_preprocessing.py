"""
Legal Text Decoder - Data Preprocessing
=======================================
Script for loading, cleaning, and preparing data for training.

This script:
1. Loads training data (budapestgo_aszf.json)
2. Loads test data (consensus/*.json)
3. Cleans and validates the data
4. Extracts text complexity features
5. Saves processed data for training
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

from config import (
    TRAIN_DATA_FILE, CONSENSUS_DIR, PROCESSED_DATA_DIR,
    RATING_MIN, RATING_MAX, RANDOM_SEED, MIN_ANNOTATIONS_PER_FILE
)
from utils import (
    setup_logger, load_json_annotations, load_all_consensus_files,
    extract_features_dataframe
)

# Setup logger
logger = setup_logger()


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Parameters:
    -----------
    text : str
        Raw text
    
    Returns:
    --------
    str
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_rating(rating: int) -> bool:
    """Check if rating is within valid range."""
    return RATING_MIN <= rating <= RATING_MAX


def clean_dataframe(df: pd.DataFrame, remove_text_duplicates: bool = False) -> pd.DataFrame:
    """Clean DataFrame by removing invalid entries."""
    original_len = len(df)
    
    # Clean text (normalize whitespace)
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    # Remove invalid ratings
    df = df[df['rating'].apply(validate_rating)]
    
    # Only remove duplicates if explicitly requested (e.g., for training data)
    if remove_text_duplicates:
        df = df.drop_duplicates(subset=['text'], keep='first')
    
    removed = original_len - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} invalid entries ({original_len} -> {len(df)})")
    
    return df.reset_index(drop=True)


# def analyze_data_distribution(df: pd.DataFrame, name: str):
#     """
#     Analyze and log data distribution.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Dataset name for logging
#     """
#     print_separator()
#     log_and_print(f"DATA DISTRIBUTION: {name}")
#     print_separator("-")
    
#     log_and_print(f"Total samples: {len(df)}")
    
#     # Rating distribution
#     log_and_print("\nRating distribution:")
#     rating_counts = df['rating'].value_counts().sort_index()
#     for rating, count in rating_counts.items():
#         percentage = (count / len(df)) * 100
#         bar = "█" * int(percentage / 2)
#         log_and_print(f"  Rating {rating}: {count:4d} ({percentage:5.1f}%) {bar}")
    
#     # Text length statistics
#     df['text_length'] = df['text'].str.len()
#     log_and_print(f"\nText length statistics:")
#     log_and_print(f"  Mean: {df['text_length'].mean():.1f} chars")
#     log_and_print(f"  Std:  {df['text_length'].std():.1f} chars")
#     log_and_print(f"  Min:  {df['text_length'].min()} chars")
#     log_and_print(f"  Max:  {df['text_length'].max()} chars")
    
#     # Word count statistics
#     df['word_count'] = df['text'].str.split().str.len()
#     log_and_print(f"\nWord count statistics:")
#     log_and_print(f"  Mean: {df['word_count'].mean():.1f} words")
#     log_and_print(f"  Std:  {df['word_count'].std():.1f} words")
#     log_and_print(f"  Min:  {df['word_count'].min()} words")
#     log_and_print(f"  Max:  {df['word_count'].max()} words")


def main():
    """Main data preprocessing pipeline."""
    logger.info("=" * 70)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 70)
    
    # Create output directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Training Data
    # =========================================================================
    logger.info("[STEP 1] Loading training data...")
    
    train_df = pd.DataFrame(columns=['text', 'rating'])
    if TRAIN_DATA_FILE.exists():
        train_df = load_json_annotations(TRAIN_DATA_FILE)
        if train_df is not None and len(train_df) > 0:
            logger.info(f"  Loaded {len(train_df)} samples from {TRAIN_DATA_FILE.name}")
        else:
            logger.info("  Training file is empty")
            train_df = pd.DataFrame(columns=['text', 'rating'])
    else:
        logger.info(f"  Training file not found: {TRAIN_DATA_FILE}")
    
    # =========================================================================
    # Step 2: Load Consensus (Test) Data
    # =========================================================================
    logger.info("[STEP 2] Loading consensus (test) data...")
    logger.info(f"  Minimum annotations per file: {MIN_ANNOTATIONS_PER_FILE}")
    
    consensus_df = pd.DataFrame(columns=['text', 'rating', 'source_file'])
    if CONSENSUS_DIR.exists():
        consensus_df = load_all_consensus_files(CONSENSUS_DIR)
        logger.info(f"  Loaded {len(consensus_df)} annotations from consensus files")
    else:
        logger.info(f"  Consensus directory not found: {CONSENSUS_DIR}")
    
    # =========================================================================
    # Step 3: Train/Test Split
    # =========================================================================
    logger.info("[STEP 3] Preparing train/test split...")
    
    if len(train_df) == 0 and len(consensus_df) > 0:
        # Split consensus data 70/30
        np.random.seed(RANDOM_SEED)
        unique_texts = consensus_df.drop_duplicates(subset=['text'])
        n_train = int(len(unique_texts) * 0.7)
        shuffled = unique_texts.sample(frac=1, random_state=RANDOM_SEED)
        train_texts = set(shuffled['text'].iloc[:n_train])
        train_df = consensus_df[consensus_df['text'].isin(train_texts)].copy()
        test_df = consensus_df[~consensus_df['text'].isin(train_texts)].copy()
        logger.info(f"  Split consensus data: {len(train_df)} train, {len(test_df)} test")
    else:
        test_df = consensus_df.copy()
        logger.info(f"  Using budapestgo_aszf.json for training, consensus for testing")
    
    # =========================================================================
    # Step 4: Clean Data
    # =========================================================================
    logger.info("[STEP 4] Cleaning data...")
    
    train_df = clean_dataframe(train_df, remove_text_duplicates=True)
    test_df = clean_dataframe(test_df, remove_text_duplicates=False)
    logger.info(f"  Train samples after cleaning: {len(train_df)}")
    logger.info(f"  Test samples after cleaning: {len(test_df)}")
    
    # =========================================================================
    # Step 5: Extract Features
    # =========================================================================
    logger.info("[STEP 5] Extracting text features...")
    
    if len(train_df) > 0:
        train_features = extract_features_dataframe(train_df)
        train_df = pd.concat([train_df.reset_index(drop=True), train_features], axis=1)
    
    if len(test_df) > 0:
        test_features = extract_features_dataframe(test_df)
        test_df = pd.concat([test_df.reset_index(drop=True), test_features], axis=1)
    
    n_features = len(train_features.columns) if len(train_df) > 0 else 0
    logger.info(f"  Extracted {n_features} features")
    
    # =========================================================================
    # Step 6: Save Processed Data
    # =========================================================================
    logger.info("[STEP 6] Saving processed data...")
    
    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    logger.info(f"  Saved: {train_path}")
    logger.info(f"  Saved: {test_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 70)
    logger.info("DATA PREPROCESSING COMPLETED")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    logger.info(f"  Features: {n_features}")
    logger.info("=" * 70)
    
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = main()
