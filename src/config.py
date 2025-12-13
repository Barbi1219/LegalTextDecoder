"""
Legal Text Decoder - Configuration File
========================================
Contains all hyperparameters, paths, and configuration settings.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Output directory (minden eredmény ide kerül)
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"

# Log directory (run.log)
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / "run.log"

# Data paths (konténeren belül, nem mountolt)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Training data (will be downloaded from SharePoint)
TRAIN_DATA_FILE = RAW_DATA_DIR / "budapestgo_aszf.json"  # Your Neptun folder file

# Test data
CONSENSUS_DIR = RAW_DATA_DIR / "consensus"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# Training configuration
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
RANDOM_SEED = 42

# Model configuration
HIDDEN_DIM = 64
DROPOUT_RATE = 0.5
NUM_CLASSES = 5  # Rating scale 1-5

# Text feature configuration
MAX_SENTENCE_LENGTH = 512
MIN_WORD_LENGTH = 2

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
# Rating scale
RATING_MIN = 1
RATING_MAX = 5
RATING_LABELS = ['1-Nagyon nehezen érthető', '2-Nehezen érthető', 
                 '3-Többé/kevésbé megértem', '4-Érthető', '5-Könnyen érthető']

MIN_ANNOTATIONS_PER_FILE = 40  # Minimum annotations required per annotator file

# Cross-validation
CV_FOLDS = 5

# =============================================================================
# TEXT COMPLEXITY FEATURES
# =============================================================================
# Feature extraction settings
# Based on correlation analysis: char_count (-0.33) and word_count (-0.34) are strongest
FEATURE_NAMES = [
    # Strong predictors (|correlation| > 0.25)
    'char_count',           # Character count (correlation: -0.33)
    'word_count',           # Word count (correlation: -0.34)
    
    # Sentence-level features
    'sentence_count',       # Number of sentences
    'avg_word_length',      # Average word length
    'avg_sentence_length',  # Average sentence length (words per sentence)
    
    # Complexity ratios
    'long_word_ratio',      # Ratio of long words (>8 characters)
    'complex_word_ratio',   # Ratio of complex words (>10 characters)
    'lexical_diversity',    # Unique words / total words
    
    # Sentence complexity indicators
    'comma_per_sentence',   # Commas per sentence (complex structure)
    'parentheses_ratio',    # Parentheses per word (legal explanations)
    'sentence_complexity',  # Combined complexity index
    
    # Character-level features
    'punctuation_ratio',    # Punctuation characters ratio
    'number_count',         # Count of numbers (references)
    'digit_ratio',          # Digit characters ratio
    'upper_word_ratio',     # Uppercase words ratio (abbreviations)
    
    # Legal terminology
    'legal_term_count',     # Count of legal terms
    'legal_term_ratio',     # Legal terms per word
]

# Hungarian legal terminology (expanded list)
LEGAL_TERMS = [
    # Contract terms
    'szerződés', 'feltétel', 'feltételek', 'megállapodás', 'egyezmény',
    
    # Rights and obligations
    'jogosult', 'kötelezett', 'köteles', 'jogosultság', 'kötelezettség',
    
    # Liability
    'felelősség', 'kártérítés', 'kártalanítás', 'helytállás',
    
    # Service terms
    'szolgáltatás', 'díj', 'díjszabás', 'teljesítés', 'igénybevétel',
    
    # Modifications
    'módosítás', 'megszűnés', 'felmondás', 'hatálybalépés',
    
    # Legal status
    'hatályos', 'érvényes', 'érvénytelen', 'hatálya', 'hatályon',
    
    # Legal references
    'törvény', 'rendelet', 'paragrafus', 'bekezdés', 'pont',
    
    # Conditional terms
    'amennyiben', 'kivéve', 'feltéve', 'abban az esetben',
    
    # Legal language
    'vonatkozó', 'tekintetében', 'alapján', 'értelmében', 'szerint',
    'rendelkezés', 'előírás', 'szabály', 'szabályozás',
    
    # Procedures
    'eljárás', 'igénybe', 'elmulasztás', 'késedelem', 'határidő',
    
    # Exclusivity
    'kizárólag', 'kizárólagos', 'korlátozás', 'tilalom',
    
    # Authorities
    'illetékes', 'hatóság', 'bíróság', 'jogvita', 'vitarendezés',
    
    # Data protection (GDPR related)
    'adatkezelés', 'személyes', 'hozzájárulás', 'tájékoztatás', 'érintett',
    'adatvédelem', 'adatfeldolgozás'
]

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# DOCKER CONFIGURATION
# =============================================================================
# Data mount point in Docker container
DOCKER_DATA_MOUNT = "/app/data"
