#!/bin/bash

echo "========================================================================"
echo "Legal Text Decoder - Complete Pipeline"
echo "========================================================================"

# Exit on error
set -e

# Step 0: Download data
echo ""
echo "[STEP 0] Downloading dataset..."
python src/00_download_data.py

# Step 1: Data preprocessing
echo ""
echo "[STEP 1] Data preprocessing..."
python src/01_data_preprocessing.py

# Step 2: Model training
echo ""
echo "[STEP 2] Model training..."
python src/02_training.py

# Step 3: Model evaluation
echo ""
echo "[STEP 3] Model evaluation..."
python src/03_evaluation.py

# Step 4: Inference demo
echo ""
echo "[STEP 4] Inference demonstration..."
python src/04_inference.py

echo ""
echo "========================================================================"
echo "Pipeline completed successfully!"
echo "========================================================================"