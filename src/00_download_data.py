"""
Legal Text Decoder - Data Download
===================================
Script for downloading and preparing the dataset from SharePoint.

This script:
1. Downloads the zip file from BME SharePoint
2. Extracts training data (LXXAMS folder)
3. Extracts test data (consensus folder)
4. Organizes files into data/raw/ structure
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import requests
import zipfile
import shutil
from typing import Optional

from config import RAW_DATA_DIR, TRAIN_DATA_FILE, CONSENSUS_DIR
from utils import setup_logger

# Setup logger
logger = setup_logger()

# SharePoint download URL
SHAREPOINT_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"

# Expected folders in zip
WRAPPER_FOLDER = "legaltextdecoder"
NEPTUN_CODE = "LXXAMS"  # Your Neptun code folder
CONSENSUS_FOLDER = "consensus"


def download_file(url: str, destination: Path) -> bool:
    """
    Download file from URL.
    
    Parameters:
    -----------
    url : str
        Download URL
    destination : Path
        Where to save the file
    
    Returns:
    --------
    bool
        True if successful
    """
    try:
        logger.info(f"Downloading data from SharePoint...")
        logger.info(f"  URL: {url[:80]}...")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"  File size: {total_size / (1024*1024):.1f} MB")
        
        # Download with progress
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 10%
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if progress % 10 < 0.1:  # Approximate 10% increments
                            logger.info(f"  Downloaded: {progress:.0f}%")
        
        logger.info(f"  Download complete: {destination.name}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_training_data(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract training data from zip (LXXAMS folder).
    
    Parameters:
    -----------
    zip_path : Path
        Path to zip file
    extract_to : Path
        Destination directory
    
    Returns:
    --------
    bool
        True if successful
    """
    try:
        logger.info(f"Extracting training data (Neptun: {NEPTUN_CODE})...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find LXXAMS folder contents
            neptun_files = [name for name in zip_ref.namelist() 
                           if name.startswith(f"{WRAPPER_FOLDER}/{NEPTUN_CODE}/") and name.endswith('.json')]
            
            if not neptun_files:
                logger.warning(f"  No JSON files found in {NEPTUN_CODE}/ folder")
                # Debug: show what folders exist
                all_folders = set(name.split('/')[0] for name in zip_ref.namelist() if '/' in name)
                logger.warning(f"  Available folders in zip: {sorted(all_folders)}")
                # Show second-level folders
                if all_folders:
                    for folder in all_folders:
                        subfolders = set(name.split('/')[1] for name in zip_ref.namelist() 
                                       if name.startswith(f"{folder}/") and name.count('/') >= 2)
                        if subfolders:
                            logger.warning(f"  Subfolders in {folder}/: {sorted(subfolders)}")
                return False
            
            logger.info(f"  Found {len(neptun_files)} JSON file(s) in {NEPTUN_CODE}/")
            
            # Extract JSON files
            for file_name in neptun_files:
                # Extract to temporary location
                zip_ref.extract(file_name, extract_to)
                
                # Move to data/raw/ (flatten directory structure)
                source = extract_to / file_name
                dest_name = Path(file_name).name
                destination = RAW_DATA_DIR / dest_name
                
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                
                logger.info(f"    Extracted: {dest_name}")
                
                # Special log for training file
                if dest_name == "budapestgo_aszf.json":
                    logger.info(f"     Training file confirmed: {destination}")
            
            # Clean up temporary Neptun folder
            neptun_folder = extract_to / NEPTUN_CODE
            if neptun_folder.exists():
                shutil.rmtree(neptun_folder)
            
            logger.info(f"  Training data extracted successfully")
            return True
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def extract_consensus_data(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract test data from zip (consensus folder).
    
    Parameters:
    -----------
    zip_path : Path
        Path to zip file
    extract_to : Path
        Destination directory
    
    Returns:
    --------
    bool
        True if successful
    """
    try:
        logger.info(f"Extracting consensus test data...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find consensus folder contents (with wrapper folder)
            consensus_files = [name for name in zip_ref.namelist() 
                              if f"{WRAPPER_FOLDER}/{CONSENSUS_FOLDER}/" in name and name.endswith('.json')]
            
            if not consensus_files:
                logger.warning(f"  No JSON files found in {WRAPPER_FOLDER}/{CONSENSUS_FOLDER}/ folder")
                return False
            
            logger.info(f"  Found {len(consensus_files)} JSON file(s) in {WRAPPER_FOLDER}/{CONSENSUS_FOLDER}/")
            
            # Create consensus directory
            CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Extract JSON files (with progress every 10 files)
            for idx, file_name in enumerate(consensus_files, 1):
                # Extract to temporary location
                zip_ref.extract(file_name, extract_to)
                
                # Move to data/raw/consensus/
                source = extract_to / file_name
                dest_name = Path(file_name).name
                destination = CONSENSUS_DIR / dest_name
                
                shutil.move(str(source), str(destination))
                
                # Log progress
                if idx % 10 == 0 or idx == len(consensus_files):
                    logger.info(f"    Extracted {idx}/{len(consensus_files)} files...")
            
            # Clean up temporary folders
            wrapper_folder = extract_to / WRAPPER_FOLDER
            if wrapper_folder.exists():
                shutil.rmtree(wrapper_folder)
            
            logger.info(f"  Consensus data extracted successfully")
            return True
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_data_exists() -> bool:
    """
    Check if data has already been downloaded.
    
    Returns:
    --------
    bool
        True if data exists
    """
    # Check for training data
    train_files = list(RAW_DATA_DIR.glob("*.json"))
    
    # Check for consensus data
    consensus_files = list(CONSENSUS_DIR.glob("*.json")) if CONSENSUS_DIR.exists() else []
    
    has_train = len(train_files) > 0
    has_consensus = len(consensus_files) > 0
    
    if has_train and has_consensus:
        logger.info("Data already exists:")
        logger.info(f"  Training files: {len(train_files)}")
        logger.info(f"  Consensus files: {len(consensus_files)}")
        return True
    
    return False


def main():
    """Main data download pipeline."""
    logger.info("=" * 70)
    logger.info("DATA DOWNLOAD")
    logger.info("=" * 70)
    
    # Check if data already exists
    if check_data_exists():
        logger.info("\nSkipping download (data already present)")
        logger.info("To re-download, delete the data/raw/ folder")
        logger.info("=" * 70)
        return True
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Temporary paths
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = temp_dir / "dataset.zip"
    
    try:
        # Step 1: Download zip
        logger.info("\n[STEP 1] Downloading dataset...")
        if not download_file(SHAREPOINT_URL, zip_path):
            logger.error("Download failed!")
            return False
        
        # Step 2: Extract training data
        logger.info("\n[STEP 2] Extracting training data...")
        if not extract_training_data(zip_path, temp_dir):
            logger.error("Training data extraction failed!")
            return False
        
        # Step 3: Extract consensus data
        logger.info("\n[STEP 3] Extracting consensus data...")
        if not extract_consensus_data(zip_path, temp_dir):
            logger.error("Consensus data extraction failed!")
            return False
        
        # Step 4: Verify
        logger.info("\n[STEP 4] Verifying data...")
        train_files = list(RAW_DATA_DIR.glob("*.json"))
        consensus_files = list(CONSENSUS_DIR.glob("*.json"))
        
        logger.info(f"  Training files: {len(train_files)}")
        logger.info(f"  Consensus files: {len(consensus_files)}")
        
        if len(train_files) == 0:
            logger.error("No training files found!")
            return False
        
        if len(consensus_files) == 0:
            logger.warning("No consensus files found!")
        
        logger.info("\n" + "=" * 70)
        logger.info("DATA DOWNLOAD COMPLETED")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        return False
        
    finally:
        # Clean up temporary files
        if zip_path.exists():
            zip_path.unlink()
            logger.info(f"Cleaned up: {zip_path}")
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up: {temp_dir}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)