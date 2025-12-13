"""
Legal Text Decoder - Utility Functions
======================================
Contains helper functions for logging, data loading, and common operations.
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from config import (
    LOG_FILE, LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
    LEGAL_TERMS, FEATURE_NAMES
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(name: str = "legal_text_decoder") -> logging.Logger:
    """
    Set up logger that outputs to stdout only.

    Docker will capture stdout and redirect to log file:
    docker run ... > log/run.log 2>&1

    Parameters:
    -----------
    name : str
        Logger name

    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler only (Docker captures this)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Simple format without timestamps (cleaner logs)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

# Initialize default logger
logger = setup_logger()

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_all_consensus_files(consensus_dir: Path, min_annotations: int = None) -> pd.DataFrame:
    """
    Load all consensus JSON files from directory.

    Parameters:
    -----------
    consensus_dir : Path
        Directory containing consensus JSON files
    min_annotations : int, optional
        Minimum number of annotations required per file (default from config)

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all annotations
    """
    from config import MIN_ANNOTATIONS_PER_FILE

    if min_annotations is None:
        min_annotations = MIN_ANNOTATIONS_PER_FILE

    all_data = []
    json_files = list(consensus_dir.glob("*.json"))

    # Filter out known non-annotation files
    json_files = [f for f in json_files if 'legaltextdecoder' not in f.name.lower()]

    logger.info(f"Found {len(json_files)} JSON files in consensus directory")

    loaded_count = 0
    skipped_empty = 0
    skipped_error = 0
    skipped_insufficient = 0

    for file_path in json_files:
        df = load_json_annotations(file_path)

        if df is None:
            #logger.warning(f"Skipping {file_path.name}: JSON decode error")
            skipped_error += 1
            continue

        if len(df) == 0:
           #logger.warning(f"Skipping {file_path.name}: empty or invalid data")
            skipped_empty += 1
            continue

        # Check minimum annotations threshold
        if len(df) < min_annotations:
            # logger.warning(
            #     f"Skipping {file_path.name}: only {len(df)} annotations "
            #     f"(minimum required: {min_annotations})"
            # )
            skipped_insufficient += 1
            continue

        df['source_file'] = file_path.stem
        all_data.append(df)
        loaded_count += 1
        #logger.info(f"   {file_path.name}: {len(df)} annotations")

    # Log summary
    logger.info(f"\nConsensus loading summary:")
    logger.info(f"  Skipped (empty or invalid):     {skipped_empty + skipped_error} files")
    logger.info(f"  Skipped (<{min_annotations} annotations): {skipped_insufficient} files")
    logger.info(f"  Valid:     {loaded_count} files")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total consensus annotations loaded: {len(combined_df)}")
        return combined_df
    else:
        logger.warning("No valid consensus data found!")
        return pd.DataFrame(columns=['text', 'rating', 'source_file'])


def load_json_annotations(file_path: Path) -> pd.DataFrame:
    """
    Load annotations from a Label Studio JSON file.

    Parameters:
    -----------
    file_path : Path
        Path to JSON file

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with text and rating columns, or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Handle empty files
            if not content:
                return pd.DataFrame(columns=['text', 'rating'])

            data = json.loads(content)

    except json.JSONDecodeError:
        return None
    except Exception:
        return None

    # Handle empty JSON array
    # if not data:
    #     return pd.DataFrame(columns=['text', 'rating'])

    filtered_data = []

    for item in data:
        try:
            # Extract text from 'data' field
            text = item.get('data', {}).get('text', '')

            # Extract rating from annotations
            rating = None
            if 'annotations' in item and len(item['annotations']) > 0:
                annotations = item['annotations'][0]
                if 'result' in annotations and len(annotations['result']) > 0:
                    result = annotations['result'][0]
                    if 'value' in result and 'choices' in result['value']:
                        choices = result['value']['choices']
                        if choices and len(choices) > 0:
                            rating_str = choices[0]
                            # Extract first digit from rating string
                            if rating_str and rating_str[0].isdigit():
                                rating = int(rating_str[0])

            # Only include entries with valid text and rating
            if rating is not None and text:
                filtered_data.append({
                    'text': text,
                    'rating': rating
                })

        except Exception as e:
            # Skip malformed entries silently
            continue

    return pd.DataFrame(filtered_data) #if filtered_data else pd.DataFrame(columns=['text', 'rating'])


# =============================================================================
# TEXT FEATURE EXTRACTION
# =============================================================================

def count_sentences(text: str) -> int:
    """Count sentences based on punctuation."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def count_legal_terms(text: str) -> int:
    """Count occurrences of legal terminology."""
    text_lower = text.lower()
    count = 0
    for term in LEGAL_TERMS:
        count += text_lower.count(term.lower())
    return count


def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract text complexity features from a single text.

    Features are designed based on correlation analysis:
    - char_count and word_count have strongest correlation with rating (-0.33, -0.34)
    - Additional features capture legal text complexity

    Parameters:
    -----------
    text : str
        Input text

    Returns:
    --------
    Dict[str, float]
        Dictionary of feature names and values
    """
    import re

    if not text or not isinstance(text, str):
        return {name: 0.0 for name in FEATURE_NAMES}

    # ===========================================
    # BASIC COUNTS (strongest predictors)
    # ===========================================
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?;]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Avoid division by zero
    if word_count == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    # ===========================================
    # WORD-LEVEL FEATURES
    # ===========================================
    word_lengths = [len(w) for w in words]
    avg_word_length = np.mean(word_lengths)
    avg_sentence_length = word_count / sentence_count

    # Long words (threshold: 8 chars, increased from 6)
    long_words = [w for w in words if len(w) > 8]
    long_word_ratio = len(long_words) / word_count

    # Complex words (>10 characters)
    complex_words = [w for w in words if len(w) > 10]
    complex_word_ratio = len(complex_words) / word_count

    # Lexical diversity (unique words ratio)
    unique_words = set(w.lower() for w in words)
    lexical_diversity = len(unique_words) / word_count

    # ===========================================
    # SENTENCE COMPLEXITY FEATURES
    # ===========================================
    # Commas per sentence (indicates complex sentence structure)
    comma_count = text.count(',')
    comma_per_sentence = comma_count / sentence_count

    # Parentheses ratio (legal texts often have explanations in parentheses)
    parentheses_count = text.count('(') + text.count(')')
    parentheses_ratio = parentheses_count / word_count

    # Combined sentence complexity index
    sentence_complexity = avg_sentence_length * (1 + comma_per_sentence / 5)

    # ===========================================
    # CHARACTER-LEVEL FEATURES
    # ===========================================
    # Punctuation
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0

    # Numbers (references, dates, amounts)
    numbers = re.findall(r'\d+', text)
    number_count = len(numbers)
    digit_ratio = len(re.findall(r'\d', text)) / char_count if char_count > 0 else 0

    # Uppercase words (abbreviations, proper nouns)
    upper_words = [w for w in words if w.isupper() and len(w) > 1]
    upper_word_ratio = len(upper_words) / word_count

    # ===========================================
    # LEGAL TERMINOLOGY
    # ===========================================
    legal_term_count = count_legal_terms(text)
    legal_term_ratio = legal_term_count / word_count

    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'long_word_ratio': long_word_ratio,
        'complex_word_ratio': complex_word_ratio,
        'lexical_diversity': lexical_diversity,
        'comma_per_sentence': comma_per_sentence,
        'parentheses_ratio': parentheses_ratio,
        'sentence_complexity': sentence_complexity,
        'punctuation_ratio': punctuation_ratio,
        'number_count': number_count,
        'digit_ratio': digit_ratio,
        'upper_word_ratio': upper_word_ratio,
        'legal_term_count': legal_term_count,
        'legal_term_ratio': legal_term_ratio,
    }


def extract_features_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Extract features for all texts in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with text column
    text_column : str
        Name of text column

    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted features
    """
    features_list = []

    for text in df[text_column]:
        features = extract_text_features(text)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    return features_df


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def log_and_print(message: str):
    """Log message to stdout (captured by Docker)."""
    logger.info(message)


def print_separator(char: str = "=", length: int = 70):
    """Print separator line."""
    logger.info(char * length)



def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as string."""
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4f}")
        else:
            lines.append(f"  {name}: {value}")
    return "\n".join(lines)


# =============================================================================
# MODEL HELPERS
# =============================================================================

def save_model(model: Any, path: Path):
    """Save model to file."""
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> Any:
    """Load model from file."""
    import pickle
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model
