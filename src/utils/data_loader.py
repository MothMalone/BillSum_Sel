"""
Custom dataset loader for BillSum Knowledge Distillation experiments.
Handles the MothMalone/SLMS-KD-Benchmarks dataset with proper authentication.
"""

import os
from datasets import load_dataset, Dataset
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BillSumLoader:
    """Custom loader for your BillSum dataset."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.dataset_path = "MothMalone/SLMS-KD-Benchmarks"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_datasets(self, max_train: Optional[int] = None) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train, test, and ca_test splits with robust error handling."""
        logger.info(f"Loading BillSum from {self.dataset_path}")
        
        try:
            # Load with authentication using newer token parameter
            auth_token = os.getenv("HF_TOKEN")
            if not auth_token:
                logger.warning("No HF_TOKEN found, trying without authentication")
                auth_token = None
            
            # Try with newer 'token' parameter first
            try:
                dataset = load_dataset(
                    self.dataset_path, 
                    name="billsum",
                    cache_dir=self.cache_dir,
                    token=auth_token
                )
            except (TypeError, ValueError) as e:
                # Fallback to older 'use_auth_token' parameter if needed
                logger.warning(f"Newer 'token' parameter failed: {e}")
                logger.info("Trying with legacy 'use_auth_token' parameter...")
                dataset = load_dataset(
                    self.dataset_path, 
                    name="billsum",
                    cache_dir=self.cache_dir,
                    use_auth_token=auth_token
                )
            
            # Validate splits exist
            available_splits = list(dataset.keys())
            logger.info(f"Available splits: {available_splits}")
            
            # Get splits with fallbacks
            train_data = dataset.get("train")
            test_data = dataset.get("test")
            ca_test_data = dataset.get("ca_test")
            
            if train_data is None:
                raise ValueError("No 'train' split found in dataset")
            if test_data is None:
                logger.warning("No 'test' split found, using train split for testing")
                test_data = train_data.train_test_split(test_size=0.1, seed=42)["test"]
            if ca_test_data is None:
                logger.warning("No 'ca_test' split found, using test split")
                ca_test_data = test_data
            
            # Validate data format
            sample = train_data[0]
            required_keys = ['text', 'summary']
            
            # Handle different column names
            if 'summary' not in sample and 'sum' in sample:
                logger.info("Found 'sum' column, renaming to 'summary'")
                train_data = train_data.rename_column('sum', 'summary')
                test_data = test_data.rename_column('sum', 'summary') 
                ca_test_data = ca_test_data.rename_column('sum', 'summary')
            
            # Validate required columns exist
            sample = train_data[0]
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Required column '{key}' not found. Available: {list(sample.keys())}")
            
            # Limit training data if requested
            if max_train and len(train_data) > max_train:
                train_data = train_data.shuffle(seed=42).select(range(max_train))
                logger.info(f"Limited training data to {max_train} samples")
            
            logger.info(f"Successfully loaded - Train: {len(train_data)}, Test: {len(test_data)}, CA_Test: {len(ca_test_data)}")
            return train_data, test_data, ca_test_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            logger.error("This might be due to:")
            logger.error("1. Missing or invalid HF_TOKEN")
            logger.error("2. Network connectivity issues")
            logger.error("3. Dataset access restrictions")
            logger.error("4. Incorrect dataset name or split")
            raise
    
    def get_dataset_stats(self) -> dict:
        """Quick dataset statistics."""
        train_data, test_data, ca_test_data = self.load_datasets()
        
        # Calculate text length statistics
        train_lengths = [len(text.split()) for text in train_data['text']]
        
        return {
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'ca_test_samples': len(ca_test_data),
            'avg_text_length': sum(train_lengths) / len(train_lengths),
            'min_text_length': min(train_lengths),
            'max_text_length': max(train_lengths),
            'avg_summary_length': sum(len(s.split()) for s in train_data['summary']) / len(train_data)
        }
    
    def validate_dataset_format(self) -> bool:
        """Validate that the dataset has the expected format."""
        try:
            train_data, _, _ = self.load_datasets()
            
            # Check required columns
            required_columns = ['text', 'summary']
            for col in required_columns:
                if col not in train_data.column_names:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            # Check data types
            sample = train_data[0]
            if not isinstance(sample['text'], str) or not isinstance(sample['summary'], str):
                logger.error("Text and summary must be strings")
                return False
            
            logger.info("Dataset format validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False
