"""
Fast heuristic-based selection methods for knowledge distillation.
Implements length and diversity-based selection strategies.
"""

import numpy as np
from datasets import Dataset
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import NLTK, install if needed
try:
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK not available, falling back to simple tokenization")
    def word_tokenize(text):
        return text.lower().split()

class QuickHeuristicSelector:
    """Fast heuristic-based selection methods."""
    
    @staticmethod
    def select_by_optimal_length(dataset: Dataset, n_select: int, 
                               min_words: int = 200, max_words: int = 800) -> List[int]:
        """Select documents in the 'sweet spot' length range."""
        logger.info(f"Selecting {n_select} samples by optimal length ({min_words}-{max_words} words)")
        
        # Calculate word counts
        word_counts = []
        valid_indices = []
        
        for i, text in enumerate(dataset['text']):
            word_count = len(text.split())
            word_counts.append(word_count)
            
            if min_words <= word_count <= max_words:
                valid_indices.append(i)
        
        logger.info(f"Found {len(valid_indices)} samples in optimal length range")
        
        if len(valid_indices) >= n_select:
            # Randomly sample from valid indices
            selected = np.random.choice(valid_indices, n_select, replace=False)
        else:
            # Include all valid + random from remaining
            remaining = list(set(range(len(dataset))) - set(valid_indices))
            additional_needed = n_select - len(valid_indices)
            additional = np.random.choice(remaining, additional_needed, replace=False)
            selected = valid_indices + additional.tolist()
        
        return selected.tolist() if hasattr(selected, 'tolist') else selected

    @staticmethod
    def select_by_diversity(dataset: Dataset, n_select: int) -> List[int]:
        """Select samples with high lexical diversity."""
        logger.info(f"Selecting {n_select} samples by lexical diversity")
        
        diversity_scores = []
        
        for i, text in enumerate(dataset['text']):
            tokens = word_tokenize(text.lower())
            if len(tokens) == 0:
                diversity_scores.append((i, 0.0))
                continue
            
            # Type-token ratio
            unique_tokens = len(set(tokens))
            ttr = unique_tokens / len(tokens)
            diversity_scores.append((i, ttr))
        
        # Sort by diversity score (descending)
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_select indices
        return [idx for idx, score in diversity_scores[:n_select]]

    @staticmethod
    def select_length_diversity_combo(dataset: Dataset, n_select: int) -> List[int]:
        """Combine length filtering with diversity selection."""
        logger.info(f"Selecting {n_select} samples using length-diversity combination")
        
        # First filter by reasonable length (100-1200 words)
        candidate_indices = []
        for i, text in enumerate(dataset['text']):
            word_count = len(text.split())
            if 100 <= word_count <= 1200:
                candidate_indices.append(i)
        
        if len(candidate_indices) <= n_select:
            return candidate_indices
        
        # From candidates, select by diversity
        candidate_dataset = dataset.select(candidate_indices)
        diversity_scores = []
        
        for i, text in enumerate(candidate_dataset['text']):
            tokens = word_tokenize(text.lower())
            if len(tokens) > 0:
                ttr = len(set(tokens)) / len(tokens)
            else:
                ttr = 0.0
            diversity_scores.append((candidate_indices[i], ttr))
        
        # Sort and select top n_select
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, score in diversity_scores[:n_select]]
    
    @staticmethod
    def select_by_summary_length_ratio(dataset: Dataset, n_select: int,
                                     optimal_ratio_min: float = 0.05,
                                     optimal_ratio_max: float = 0.20) -> List[int]:
        """Select samples with optimal text-to-summary length ratios."""
        logger.info(f"Selecting {n_select} samples by summary-to-text ratio")
        
        ratio_scores = []
        
        for i, (text, summary) in enumerate(zip(dataset['text'], dataset['summary'])):
            text_len = len(text.split())
            summary_len = len(summary.split())
            
            if text_len == 0:
                ratio = 0.0
            else:
                ratio = summary_len / text_len
            
            # Score based on distance from optimal range
            if optimal_ratio_min <= ratio <= optimal_ratio_max:
                score = 1.0  # Perfect range
            elif ratio < optimal_ratio_min:
                score = ratio / optimal_ratio_min  # Too short summary
            else:
                score = optimal_ratio_max / ratio  # Too long summary
            
            ratio_scores.append((i, score))
        
        # Sort by score (descending)
        ratio_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, score in ratio_scores[:n_select]]
    
    @staticmethod
    def select_balanced_lengths(dataset: Dataset, n_select: int, n_bins: int = 5) -> List[int]:
        """Select samples balanced across different text length bins."""
        logger.info(f"Selecting {n_select} samples balanced across {n_bins} length bins")
        
        # Calculate text lengths
        lengths = [(i, len(text.split())) for i, text in enumerate(dataset['text'])]
        lengths.sort(key=lambda x: x[1])  # Sort by length
        
        # Create bins
        bin_size = len(lengths) // n_bins
        selected_indices = []
        samples_per_bin = n_select // n_bins
        remaining_samples = n_select % n_bins
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = (bin_idx + 1) * bin_size if bin_idx < n_bins - 1 else len(lengths)
            
            bin_samples = lengths[start_idx:end_idx]
            
            # Select from this bin
            n_from_bin = samples_per_bin
            if remaining_samples > 0:
                n_from_bin += 1
                remaining_samples -= 1
            
            n_from_bin = min(n_from_bin, len(bin_samples))
            
            if n_from_bin > 0:
                # Random selection from this bin
                selected_from_bin = np.random.choice(
                    [idx for idx, _ in bin_samples], 
                    n_from_bin, 
                    replace=False
                )
                selected_indices.extend(selected_from_bin)
        
        return selected_indices[:n_select]
    
    @staticmethod
    def select_high_information_density(dataset: Dataset, n_select: int) -> List[int]:
        """Select samples with high information density (unique words per sentence)."""
        logger.info(f"Selecting {n_select} samples by information density")
        
        density_scores = []
        
        for i, text in enumerate(dataset['text']):
            sentences = text.split('.')
            if len(sentences) == 0:
                density_scores.append((i, 0.0))
                continue
            
            total_words = len(text.split())
            unique_words = len(set(text.lower().split()))
            
            # Information density: unique words per sentence
            density = unique_words / len(sentences) if len(sentences) > 0 else 0.0
            density_scores.append((i, density))
        
        # Sort by density (descending)
        density_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, score in density_scores[:n_select]]
