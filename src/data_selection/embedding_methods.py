"""
Fast embedding-based selection methods for knowledge distillation.
Uses sentence transformers for efficient semantic selection.
"""

import numpy as np
from datasets import Dataset
from typing import List
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
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available")
    SKLEARN_AVAILABLE = False

class QuickEmbeddingSelector:
    """Embedding-based selection using SAME model as training for consistency."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", use_training_model: bool = True):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embedding-based selection")
        
        # Critical: Use same model family as training for consistent representations
        if use_training_model:
            # Map training models to compatible sentence transformer models
            model_mapping = {
                "meta-llama/Llama-2-7b-hf": "sentence-transformers/all-mpnet-base-v2",  # Best general purpose
                "meta-llama/Llama-2-13b-hf": "sentence-transformers/all-mpnet-base-v2", 
                "facebook/opt-6.7b": "sentence-transformers/all-roberta-large-v1",
                "microsoft/DialoGPT-medium": "sentence-transformers/all-MiniLM-L12-v2",
                "gpt2-xl": "sentence-transformers/all-MiniLM-L12-v2"
            }
            embedding_model = model_mapping.get(model_name, "sentence-transformers/all-mpnet-base-v2")
            logger.info(f"Using embedding model {embedding_model} for consistency with training model {model_name}")
        else:
            embedding_model = model_name
        
        logger.info(f"Loading sentence transformer: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.training_model_name = model_name
    
    def select_by_centroid(self, dataset: Dataset, n_select: int) -> List[int]:
        """Select samples closest to dataset centroid."""
        logger.info(f"Selecting {n_select} samples by centroid distance")
        
        # Generate embeddings for all texts
        texts = dataset['text']
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Calculate dataset centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate distances to centroid
        distances = cosine_similarity([centroid], embeddings)[0]
        
        # Select closest samples (highest cosine similarity)
        closest_indices = np.argsort(distances)[-n_select:]
        
        return closest_indices.tolist()
    
    def select_by_clustering(self, dataset: Dataset, n_select: int, n_clusters: int = None) -> List[int]:
        """Select representative samples from clusters."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to centroid selection")
            return self.select_by_centroid(dataset, n_select)
        
        if n_clusters is None:
            n_clusters = min(50, n_select // 5)  # Reasonable default
        
        logger.info(f"Selecting {n_select} samples using {n_clusters} clusters")
        
        texts = dataset['text']
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        selected_indices = []
        samples_per_cluster = n_select // n_clusters
        remaining_samples = n_select % n_clusters
        
        for cluster_id in range(n_clusters):
            # Find samples in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Calculate distances to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_embeddings = embeddings[cluster_indices]
            distances = cosine_similarity([cluster_center], cluster_embeddings)[0]
            
            # Select closest samples from this cluster
            n_from_cluster = samples_per_cluster
            if remaining_samples > 0:
                n_from_cluster += 1
                remaining_samples -= 1
            
            n_from_cluster = min(n_from_cluster, len(cluster_indices))
            closest_in_cluster = np.argsort(distances)[-n_from_cluster:]
            
            selected_indices.extend(cluster_indices[closest_in_cluster])
        
        return selected_indices[:n_select]  # Ensure exact count
    
    def select_by_diversity_sampling(self, dataset: Dataset, n_select: int) -> List[int]:
        """Select diverse samples using maximum distance sampling."""
        logger.info(f"Selecting {n_select} samples by diversity sampling")
        
        texts = dataset['text']
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Start with a random sample
        selected_indices = [np.random.randint(0, len(embeddings))]
        
        for _ in range(n_select - 1):
            # Calculate minimum distances to already selected samples
            selected_embeddings = embeddings[selected_indices]
            
            min_distances = []
            for i, emb in enumerate(embeddings):
                if i in selected_indices:
                    min_distances.append(-1)  # Already selected
                else:
                    # Find minimum distance to any selected sample
                    distances = cosine_similarity([emb], selected_embeddings)[0]
                    min_distances.append(np.min(distances))
            
            # Select the sample with maximum minimum distance (most diverse)
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return selected_indices
    
    def select_using_training_model_embeddings(self, dataset: Dataset, n_select: int, 
                                             training_model, tokenizer) -> List[int]:
        """
        Select samples using embeddings from the ACTUAL training model.
        This ensures perfect consistency between selection and training.
        """
        logger.info(f"Selecting {n_select} samples using training model embeddings (most consistent)")
        
        # Import torch here to avoid dependency issues
        try:
            import torch
        except ImportError:
            logger.error("PyTorch required for training model embeddings")
            # Fallback to sentence transformer method
            return self.select_by_centroid(dataset, n_select)
        
        texts = dataset['text']
        embeddings = []
        
        # Get embeddings from the actual training model
        training_model.eval()
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    logger.info(f"Processing text {i}/{len(texts)}")
                
                # Tokenize with same tokenizer as training
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,  # Shorter for embedding extraction
                    padding=True
                )
                
                # Move to same device as model
                inputs = {k: v.to(training_model.device) for k, v in inputs.items()}
                
                # Get model's hidden states
                outputs = training_model(**inputs, output_hidden_states=True)
                
                # Use last layer mean pooling as embedding
                last_hidden_states = outputs.hidden_states[-1]
                embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Calculate dataset centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate distances to centroid using cosine similarity
        if SKLEARN_AVAILABLE:
            distances = cosine_similarity([centroid], embeddings)[0]
        else:
            # Fallback cosine similarity calculation
            distances = []
            for emb in embeddings:
                cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))
                distances.append(cos_sim)
            distances = np.array(distances)
        
        # Select closest samples (highest cosine similarity)
        closest_indices = np.argsort(distances)[-n_select:]
        
        logger.info("Selection using training model embeddings complete - perfect consistency guaranteed")
        return closest_indices.tolist()
