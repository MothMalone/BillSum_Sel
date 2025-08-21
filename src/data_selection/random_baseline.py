"""
Random baseline selector for knowledge distillation experiments.
Provides simple random sampling as a baseline for comparison.
"""

import random
import numpy as np
from datasets import Dataset
from typing import List
import logging

logger = logging.getLogger(__name__)

class RandomSelector:
    """Random baseline selector."""
    
    @staticmethod
    def select_random(dataset: Dataset, n_select: int, seed: int = 42) -> List[int]:
        """
        Select n_select random samples from the dataset.
        
        Args:
            dataset: The source dataset
            n_select: Number of samples to select
            seed: Random seed for reproducibility
            
        Returns:
            List of selected indices
        """
        random.seed(seed)
        np.random.seed(seed)
        
        total_samples = len(dataset)
        if n_select >= total_samples:
            logger.warning(f"Requested {n_select} samples but dataset only has {total_samples}")
            return list(range(total_samples))
        
        selected_indices = np.random.choice(total_samples, n_select, replace=False)
        
        logger.info(f"Randomly selected {n_select} samples from {total_samples} total")
        return selected_indices.tolist()
    
    @staticmethod
    def select_stratified_random(dataset: Dataset, n_select: int, 
                               length_bins: int = 5, seed: int = 42) -> List[int]:
        """
        Select random samples stratified by text length.
        
        Args:
            dataset: The source dataset
            n_select: Number of samples to select
            length_bins: Number of length-based strata
            seed: Random seed for reproducibility
            
        Returns:
            List of selected indices
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Calculate text lengths
        lengths = [len(text.split()) for text in dataset['text']]
        
        # Create length-based bins
        length_percentiles = np.percentile(lengths, np.linspace(0, 100, length_bins + 1))
        
        # Assign each sample to a bin
        bin_assignments = np.digitize(lengths, length_percentiles[1:-1])
        
        # Select proportionally from each bin
        selected_indices = []
        samples_per_bin = n_select // length_bins
        remaining_samples = n_select % length_bins
        
        for bin_id in range(length_bins):
            bin_indices = [i for i, bin_val in enumerate(bin_assignments) if bin_val == bin_id]
            
            if not bin_indices:
                continue
            
            # Add extra sample to some bins if remainder exists
            n_from_bin = samples_per_bin
            if remaining_samples > 0:
                n_from_bin += 1
                remaining_samples -= 1
            
            n_from_bin = min(n_from_bin, len(bin_indices))
            
            if n_from_bin > 0:
                selected_from_bin = np.random.choice(bin_indices, n_from_bin, replace=False)
                selected_indices.extend(selected_from_bin.tolist())
        
        logger.info(f"Stratified random selection: {len(selected_indices)} samples across {length_bins} bins")
        return selected_indices[:n_select]  # Ensure exact count
