"""
Iterative data selection methods for knowledge distillation.
Implements D2-pruning and other iterative selection strategies.
"""

import numpy as np
from datasets import Dataset
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available")
    SKLEARN_AVAILABLE = False

class IterativePruningSelector:
    """Iterative pruning-based selection methods."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer for embeddings."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for iterative selection")
        
        logger.info(f"Loading sentence transformer for iterative selection: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def select_by_d2_pruning(self, dataset: Dataset, n_select: int, 
                           initial_factor: float = 2.0, iterations: int = 3) -> List[int]:
        """
        D2-based iterative pruning for data selection.
        
        Starts with initial_factor * n_select samples, then iteratively
        prunes based on data density until reaching n_select samples.
        
        Args:
            dataset: Source dataset
            n_select: Final number of samples to select
            initial_factor: Initial selection factor (e.g., 2.0 = start with 2x samples)
            iterations: Number of pruning iterations
            
        Returns:
            List of selected indices
        """
        logger.info(f"D2-pruning: {n_select} samples via {iterations} iterations")
        
        # Start with larger initial set
        initial_size = min(int(n_select * initial_factor), len(dataset))
        current_indices = list(range(len(dataset)))
        
        # Generate embeddings for all samples
        texts = dataset['text']
        logger.info("Generating embeddings for D2-pruning...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Iteratively prune
        for iteration in range(iterations):
            target_size = int(initial_size * ((n_select / initial_size) ** ((iteration + 1) / iterations)))
            target_size = max(target_size, n_select)  # Don't go below final target
            
            if len(current_indices) <= target_size:
                break
            
            logger.info(f"Iteration {iteration + 1}: pruning from {len(current_indices)} to {target_size}")
            
            # Calculate data density for current set
            current_embeddings = embeddings[current_indices]
            density_scores = self._calculate_data_density(current_embeddings)
            
            # Keep samples with highest density (most representative)
            ranked_indices = sorted(enumerate(current_indices), 
                                  key=lambda x: density_scores[x[0]], reverse=True)
            current_indices = [idx for _, idx in ranked_indices[:target_size]]
        
        # Final selection to exact target
        if len(current_indices) > n_select:
            current_embeddings = embeddings[current_indices]
            density_scores = self._calculate_data_density(current_embeddings)
            ranked_indices = sorted(enumerate(current_indices), 
                                  key=lambda x: density_scores[x[0]], reverse=True)
            current_indices = [idx for _, idx in ranked_indices[:n_select]]
        
        logger.info(f"D2-pruning completed: selected {len(current_indices)} samples")
        return current_indices
    
    def _calculate_data_density(self, embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Calculate data density using k-nearest neighbors.
        
        Args:
            embeddings: Embedding matrix (n_samples, embedding_dim)
            k: Number of nearest neighbors for density calculation
            
        Returns:
            Density scores for each sample
        """
        if not SKLEARN_AVAILABLE:
            # Fallback to simple pairwise distances
            return self._calculate_simple_density(embeddings)
        
        n_samples = len(embeddings)
        k = min(k, n_samples - 1)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # For each sample, find k nearest neighbors and calculate density
        density_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get similarities to all other samples (excluding self)
            sample_similarities = similarities[i]
            sample_similarities[i] = -1  # Exclude self
            
            # Find k nearest neighbors
            nearest_k = np.argsort(sample_similarities)[-k:]
            
            # Density = average similarity to k nearest neighbors
            density_scores[i] = np.mean(sample_similarities[nearest_k])
        
        return density_scores
    
    def _calculate_simple_density(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple density calculation without sklearn."""
        n_samples = len(embeddings)
        density_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Calculate distances to all other samples
            distances = []
            for j in range(n_samples):
                if i != j:
                    # Cosine similarity
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    distances.append(sim)
            
            # Density = average similarity to all other samples
            density_scores[i] = np.mean(distances) if distances else 0.0
        
        return density_scores
    
    def select_by_progressive_filtering(self, dataset: Dataset, n_select: int,
                                     stages: List[Dict[str, Any]] = None) -> List[int]:
        """
        Progressive filtering with multiple stages.
        
        Args:
            dataset: Source dataset
            n_select: Final number of samples
            stages: List of filtering stages with criteria
            
        Returns:
            List of selected indices
        """
        if stages is None:
            # Default progressive filtering stages
            stages = [
                {"filter": "length", "min_words": 50, "max_words": 2000},
                {"filter": "diversity", "top_percentile": 0.3},
                {"filter": "embedding", "method": "centroid"}
            ]
        
        logger.info(f"Progressive filtering with {len(stages)} stages to select {n_select} samples")
        
        current_indices = list(range(len(dataset)))
        
        for i, stage in enumerate(stages):
            logger.info(f"Stage {i+1}: {stage}")
            
            if stage["filter"] == "length":
                current_indices = self._filter_by_length(
                    dataset, current_indices, 
                    stage.get("min_words", 50), 
                    stage.get("max_words", 2000)
                )
            
            elif stage["filter"] == "diversity":
                target_size = max(n_select, int(len(current_indices) * stage.get("top_percentile", 0.3)))
                current_indices = self._filter_by_diversity(
                    dataset, current_indices, target_size
                )
            
            elif stage["filter"] == "embedding":
                if stage.get("method") == "centroid":
                    current_indices = self._filter_by_centroid(
                        dataset, current_indices, min(n_select * 2, len(current_indices))
                    )
            
            logger.info(f"After stage {i+1}: {len(current_indices)} samples remaining")
            
            if len(current_indices) <= n_select:
                break
        
        # Final selection
        if len(current_indices) > n_select:
            # Random selection from remaining
            current_indices = np.random.choice(current_indices, n_select, replace=False).tolist()
        
        return current_indices
    
    def _filter_by_length(self, dataset: Dataset, indices: List[int], 
                         min_words: int, max_words: int) -> List[int]:
        """Filter by text length."""
        filtered = []
        for idx in indices:
            word_count = len(dataset['text'][idx].split())
            if min_words <= word_count <= max_words:
                filtered.append(idx)
        return filtered
    
    def _filter_by_diversity(self, dataset: Dataset, indices: List[int], 
                           target_size: int) -> List[int]:
        """Filter by lexical diversity."""
        diversity_scores = []
        
        for idx in indices:
            text = dataset['text'][idx].lower()
            tokens = text.split()
            if len(tokens) > 0:
                ttr = len(set(tokens)) / len(tokens)
            else:
                ttr = 0.0
            diversity_scores.append((idx, ttr))
        
        # Sort by diversity and take top samples
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in diversity_scores[:target_size]]
    
    def _filter_by_centroid(self, dataset: Dataset, indices: List[int], 
                          target_size: int) -> List[int]:
        """Filter by distance to semantic centroid."""
        # Get embeddings for subset
        texts = [dataset['text'][idx] for idx in indices]
        embeddings = self.model.encode(texts, batch_size=32)
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate distances
        if SKLEARN_AVAILABLE:
            distances = cosine_similarity([centroid], embeddings)[0]
        else:
            distances = [np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb)) 
                        for emb in embeddings]
        
        # Select closest to centroid
        closest_indices = np.argsort(distances)[-target_size:]
        return [indices[i] for i in closest_indices]
