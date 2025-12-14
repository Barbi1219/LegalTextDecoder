# Legal Text Decoder

**Deep Learning Class (VITMMA19) Project Work**

---

## Project Information

 **Selected Topic:**  Legal Text Decoder   
 **Student Name:**  Bodnár Barbara   
 **Aiming for +1 Mark:**  Yes  

---

## Solution Description

The Legal Text Decoder project aims to predict the comprehensibility of Terms and Conditions (ÁSZF) text passages for an average person. This is an ordinal classification/regression problem where passages are rated on a 1-5 scale:

- **1**: Very Hard to Understand
- **2**: Hard to Understand  
- **3**: Somewhat Understandable
- **4**: Understandable
- **5**: Easy to Understand

The project follows an **incremental model development** approach, implementing five models of increasing complexity:

1. **Baseline Model**: Most Frequent Class predictor
2. **Logistic Regression**: Uses 17 hand-crafted text complexity features
3. **Random Forest Classifier**: Ensemble tree-based model with feature importance analysis
4. **Gradient Boosting Classifier**: Advanced ensemble model with incremental learning
5. **Neural Network (MLP)**: PyTorch Multi-Layer Perceptron with BatchNorm and Dropout regularization

**All models except the baseline use the same 17 hand-crafted text complexity features as input.**  

**Text Complexity Features** include character/word/sentence counts, average lengths, lexical diversity, long word ratios, punctuation density, and legal terminology frequency (Hungarian legal terms).  

**Model Architectures**:
1. **Baseline**: No learning, predicts most frequent class from training data
2. **Logistic Regression**: Linear combination of 17 features + StandardScaler + L2 regularization
3. **Random Forest**: 50 decision trees with max_depth=5, min_samples_split=10, min_samples_leaf=5 + StandardScaler
4. **Gradient Boosting**: 40 trees with max_depth=5, learning_rate=0.05, min_samples_split=10, min_samples_leaf=5, subsample=0.8 + StandardScaler
5. **Neural Network (MLP)**: 
- Input layer: 17 features
- Hidden layer 1: 64 neurons + ReLU + BatchNorm + Dropout(0.5)
- Hidden layer 2: 32 neurons + ReLU + BatchNorm + Dropout(0.5)
- Output layer: 5 classes (softmax)
- Optimizer: Adam (lr=0.0005)
- Loss: CrossEntropyLoss

**Evaluation Metrics**

1. Quadratic Weighted Kappa (QWK)
- Considers the ordinal nature of ratings (1 < 2 < 3 < 4 < 5)
- Penalizes larger errors more heavily (predicting 1 when true is 5 is worse than predicting 4)
- Range: [-1, 1], where 1 = perfect agreement, 0 = random, <0 = worse than random
- QWK = 1 - (sum of weighted disagreements) / (sum of weighted chance disagreements)


2. Regression Metrics: MSE, RMSE, MAE, MAPE

3. Classification Metrics: Accuracy, Precision, Recall, F1-Macro

4. Ordinal-Specific Metrics: Adjacent Accuracy, Confusion Matrix

**Results**: All models are evaluated on consensus-labeled test data using comprehensive metrics including Quadratic Weighted Kappa (primary ordinal metric), regression metrics (MSE, RMSE, MAE), and classification metrics (Accuracy, F1). The Neural Network model achieves the best performance with QWK = 0.42, Accuracy = 40%, Adjacent Accuracy = 81% and MAE = 0.84.

---

## Extra Credit Justification

The following parts were executed in my work (extra parts are marked with '+'):

1. **Containerization**: Full Docker implementation with PyTorch CPU base image, automated pipeline execution
2. **Data Acquisition and Analysis**: Downloading datasets with urls to mounted location
3. **Data Cleansing and Preparation (+)**: Comprehensive preprocessing pipeline handling multiple annotators, consensus computation, duplicate removal, and feature engineering
4. **Evaluation Criteria Definition (+)**: Implemented specialized ordinal classification metrics (Quadratic Weighted Kappa, Adjacent Accuracy) alongside standard regression and classification metrics
5. **Baseline model**: Implemented a baseline model, which predicts based on the most frequent class in the training dataset.
6. **Incremental Model Development (+)**: Five models with progressively increasing complexity, each building upon insights from the previous
7. **Advanced Evaluation (+)**: 
   - 8+ evaluation metrics covering regression, classification, and ordinal aspects (MSE, RMSE, MAE, MAPE, Macro_MAE, Accuracy, F1_Macro, F1_Weighted, QWK, Adjacent_Accuracy)
   - Detailed error analysis with confusion matrix and distribution visualization
   - Cross-model comparison with statistical significance testing
   - Feature importance analysis for interpretability
8. **Advanced Inference**: Three-mode inference system (demo, interactive, batch) for real-world deployment

---

## Docker Instructions

This project is containerized using Docker for reproducible execution across different environments.

### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

The build uses `python:3.10-slim` as base image and installs all dependencies from `requirements.txt`.

### Run

To run the complete solution pipeline with log output:

**Command Prompt:**

```bash
docker run --rm -v %cd%/output:/app/output -v %cd%/log:/app/log -v %cd%/data:/app/data dl-project > log/run.log 2>&1
```

The `> log/run.log 2>&1` ensures that all output (stdout and stderr) is captured for submission.

**Important**: The container runs the entire pipeline by default (`src/run.sh`). This includes:
1. Data preprocessing (`01_data_preprocessing.py`)
2. Model training (`02_training.py`)
3. Model evaluation (`03_evaluation.py`)
4. Inference demonstration (`04_inference.py`)

### Run Inference Only

To run inference on new texts using the trained models:  

**Interactive mode:**
```bash
# Command Prompt
docker run -it --rm -v %cd%/output:/app/output dl-project python src/04_inference.py --interactive
```

**What happens:**
1. Container starts with the **same trained models** from the pipeline
2. Interactive prompt appears where you can enter text passages
3. Models predict comprehensibility rating (1-5)
4. Type `quit` or `exit` to stop

**Batch mode:**
```bash
# Command Prompt
docker run --rm ^
  -v %cd%\output:/app/output ^
  -v %cd%\new_texts.csv:/app/input.csv ^
  dl-project python src/04_inference.py --file /app/input.csv --output /app/output/inference_predictions.csv
```

**What happens:**
- Reads `new_texts.csv` from your host machine
- Loads trained models from `output/models/`
- Predicts rating for each text
- Saves results to `output/inference_predictions.csv` on your host machine

**Requirements:**
- First row must be header: `text`
- Text must be quoted with `"` (handles internal commas)
- Save with UTF-8 encoding

**Additional parameter**:
- `--model <model_name>`: Choose specific model (default: `gradient_boosting`)

Available models:
 - `baseline` Most Frequent Class Baseline 
 - `logistic` Logistic Regression 
 - `random_forest` Random Forest Classifier
 - `gradient_boosting` Gradient Boosting (default, best performance) 
 - `neural_network`  MLP Neural Network 

---

## File Structure and Functions

The repository is structured as follows:

```
LegalTextDecoder/
├── src/                              # Source code
│   ├── config.py                     # Configuration and hyperparameters
│   ├── utils.py                      # Utility functions and logging
│   ├── models.py                     # Model class definitions
│   │
|   |── 00_download_data.py           # Downloading and preparing the dataset from SharePoint
│   ├── 01_data_preprocessing.py      # Data loading, cleaning, and feature extraction
│   ├── 02_training.py                # Model training pipeline (5 models)
│   ├── 03_evaluation.py              # Model evaluation and metrics computation
│   ├── 04_inference.py               # Inference on new data (3 modes)
│   └── run.sh                        # Pipeline orchestration script
│
├── notebook/                         # Jupyter notebooks
│   ├── 01-data-exploration.ipynb     # Exploratory Data Analysis
│   └── 02-label-analysis.ipynb       # Label distribution analysis
│
├── log/                              # Log file 
│   └── run.log                       # Complete training and evaluation logs(created by pipeline)
│ 
├── output/                           # Output directory (all files in it are created by pipeline)
│   ├── models/                       # Saved models
│   │   ├── baseline_model.pkl        # Most Frequent Class baseline
│   │   ├── logistic_model.pkl        # Logistic Regression
│   │   ├── random_forest_model.pkl   # Random Forest
│   │   ├── gradient_boosting_model.pkl  # Gradient Boosting (best)
│   │   ├── neural_network_model.pkl  # MLP Neural Network
│   ├── predictions.csv               # Test predictions (all models)
│   ├── model_comparison.csv          # Model performance comparison
│   └── inference_predictions.csv     # Inference batch predictions
│
├── data/                             # Data directory (created by pipeline)
│   ├── raw/                          # Raw JSON annotation files
│   │   ├── budapestgo_aszf.json      # Training data
│   │   └── consensus/*.json          # Test data (annotator consensus)
│   └── processed/                    # Processed CSV files
│       ├── train_processed.csv       # Training features and labels
│       ├── train_features.csv        # Training features only
│       ├── test_processed.csv        # Test features and labels
│       └── test_features.csv         # Test features only
│
├── Dockerfile                        # Docker configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Data Preparation

### Raw Data Format

Data is stored in Label Studio annotation JSON format:

```json
{
  "data": {"text": "Az ügyfél köteles..."},
  "annotations": [{
    "result": [{
      "value": {"choices": ["3-Többé/kevésbé megértem"]}
    }]
  }]
}
```

### Data Preparation Pipeline

The dataset is **automatically downloaded** from BME SharePoint when running the Docker container. No manual data download is required!

The `00_download_data.py` script:
1. Downloads the dataset zip from SharePoint
2. Extracts training data from your Neptun folder (LXXAMS)
3. Extracts test data from consensus folder
4. Organizes files into `data/raw/` structure

**Important**: The download happens automatically on first run. Subsequent runs skip the download if data already exists.

The `01_data_preprocessing.py` script performs:

1. **Data Loading**: Reads JSON files using `load_json_annotations()`
2. **Data Cleaning**:
   - Removes empty texts
   - Filters invalid ratings (outside 1-5 range)
3. **Consensus Computation**: For test data, computes consensus labels from multiple annotators
4. **Feature Extraction**: Extracts 17 text complexity features per sample
5. **Train/Test Split**: 
   - Training: `training/budapestgo_aszf.json` (133 samples)
   - Test: `consensus/*.json` files (2,231 samples)

### Text Complexity Features (17 total)

| Feature | Description |
|---------|-------------|
| `char_count` | Total character count |
| `word_count` | Total word count |
| `sentence_count` | Total sentence count |
| `avg_word_length` | Average word length |
| `avg_sentence_length` | Average sentence length (words) |
| `long_word_ratio` | Ratio of words with >6 characters |
| `complex_word_ratio` | Ratio of words with >10 characters |
| `lexical_diversity` | Type-token ratio (unique words / total words) |
| `comma_per_sentence` | Average commas per sentence |
| `parentheses_ratio` | Parentheses count / word count |
| `sentence_complexity` | Average words per sentence |
| `punctuation_ratio` | Punctuation characters / total characters |
| `number_count` | Count of numeric values |
| `digit_ratio` | Digit characters / total characters |
| `upper_word_ratio` | Ratio of words starting with uppercase |
| `legal_term_count` | Count of Hungarian legal terminology |
| `legal_term_ratio` | Legal terms / total words |

### Running Data Preparation

```bash
python src/01_data_preprocessing.py
```

Outputs saved to `data/processed/`:
- `train_processed.csv` - Training data with features and labels
- `test_processed.csv` - Test data with features and labels
- `train_features.csv` - Training features only
- `test_features.csv` - Test features only

---