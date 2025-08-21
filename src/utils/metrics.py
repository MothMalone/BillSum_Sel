"""
Evaluation metrics utilities for knowledge distillation experiments.
Provides consistent metric calculation across all methods.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    logger.warning("rouge-score not available")
    ROUGE_AVAILABLE = False

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    logger.warning("bert-score not available")
    BERT_SCORE_AVAILABLE = False

class MetricsCalculator:
    """Unified metrics calculation for text summarization."""
    
    def __init__(self, metrics: List[str] = None, device: str = "auto"):
        """
        Initialize metrics calculator.
        
        Args:
            metrics: List of metrics to calculate ['rouge1', 'rouge2', 'rougeL', 'bert_score']
            device: Device for BERTScore computation ('auto', 'cuda', 'cpu')
        """
        if metrics is None:
            metrics = ['rouge1', 'rouge2', 'rougeL', 'bert_score']
        
        self.metrics = metrics
        self.device = device
        
        # Initialize ROUGE scorer if needed
        if any('rouge' in m for m in metrics) and ROUGE_AVAILABLE:
            rouge_types = [m for m in metrics if 'rouge' in m]
            self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def calculate_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate all configured metrics."""
        results = {}
        
        # Basic statistics
        results.update(self._calculate_basic_stats(predictions, references))
        
        # ROUGE scores
        if self.rouge_scorer:
            results.update(self._calculate_rouge_scores(predictions, references))
        
        # BERTScore
        if 'bert_score' in self.metrics and BERT_SCORE_AVAILABLE:
            results.update(self._calculate_bert_score(predictions, references))
        
        # Fallback metrics if main ones unavailable
        if not self.rouge_scorer:
            results.update(self._calculate_fallback_metrics(predictions, references))
        
        return results
    
    def _calculate_basic_stats(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate basic text statistics."""
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        return {
            'num_samples': len(predictions),
            'avg_prediction_length': np.mean(pred_lengths),
            'avg_reference_length': np.mean(ref_lengths),
            'std_prediction_length': np.std(pred_lengths),
            'std_reference_length': np.std(ref_lengths),
            'avg_length_ratio': np.mean([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)])
        }
    
    def _calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        rouge_scores = {metric: [] for metric in self.metrics if 'rouge' in metric}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric in rouge_scores.keys():
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Calculate averages and standard deviations
        results = {}
        for metric, scores in rouge_scores.items():
            results[f'{metric}_avg'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_median'] = np.median(scores)
        
        return results
    
    def _calculate_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore using DeBERTa-XLarge-MNLI for high-quality embeddings."""
        try:
            # Use DeBERTa-XLarge-MNLI for superior semantic similarity evaluation
            P, R, F1 = bert_score.score(
                predictions, 
                references, 
                model_type="microsoft/deberta-xlarge-mnli",
                lang='en', 
                verbose=False,
                device=self.device
            )
            
            logger.info("✅ BERTScore calculated using microsoft/deberta-xlarge-mnli")
            return {
                'bert_score_precision': P.mean().item(),
                'bert_score_recall': R.mean().item(), 
                'bert_score_f1': F1.mean().item(),
                'bert_score_precision_std': P.std().item(),
                'bert_score_recall_std': R.std().item(),
                'bert_score_f1_std': F1.std().item()
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return {}
    
    def _calculate_fallback_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate simple fallback metrics when ROUGE is unavailable."""
        word_overlaps = []
        char_overlaps = []
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            # Word overlap (Jaccard similarity)
            if len(ref_words) > 0:
                word_overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
                word_overlaps.append(word_overlap)
            
            # Character overlap
            pred_chars = set(pred.lower())
            ref_chars = set(ref.lower())
            if len(ref_chars) > 0:
                char_overlap = len(pred_chars & ref_chars) / len(pred_chars | ref_chars)
                char_overlaps.append(char_overlap)
            
            # Exact match
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1
        
        return {
            'word_overlap_avg': np.mean(word_overlaps) if word_overlaps else 0.0,
            'char_overlap_avg': np.mean(char_overlaps) if char_overlaps else 0.0,
            'exact_match_rate': exact_matches / len(predictions),
            'word_overlap_std': np.std(word_overlaps) if word_overlaps else 0.0
        }

class ComparisonAnalyzer:
    """Analyze and compare results across different methods."""
    
    @staticmethod
    def compare_methods(results_dict: Dict[str, Dict[str, Any]], 
                       baseline_method: str = 'random') -> Dict[str, Any]:
        """Compare methods against a baseline."""
        if baseline_method not in results_dict:
            logger.warning(f"Baseline method '{baseline_method}' not found")
            baseline_method = list(results_dict.keys())[0]
        
        baseline_results = results_dict[baseline_method]['results']
        comparison = {
            'baseline_method': baseline_method,
            'methods': {}
        }
        
        for method, data in results_dict.items():
            method_results = data['results']
            
            method_comparison = {
                'absolute_scores': method_results,
                'relative_to_baseline': {},
                'training_efficiency': {}
            }
            
            # Calculate relative improvements
            for metric, value in method_results.items():
                if metric in baseline_results and 'avg' in metric:
                    baseline_value = baseline_results[metric]
                    if baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                        method_comparison['relative_to_baseline'][metric] = improvement
            
            # Training efficiency metrics
            train_samples = data.get('train_samples', 0)
            if 'rouge1_avg' in method_results:
                rouge1_per_sample = method_results['rouge1_avg'] / train_samples if train_samples > 0 else 0
                method_comparison['training_efficiency']['rouge1_per_sample'] = rouge1_per_sample
            
            comparison['methods'][method] = method_comparison
        
        return comparison
    
    @staticmethod
    def rank_methods(results_dict: Dict[str, Dict[str, Any]], 
                    primary_metric: str = 'rougeL_avg') -> List[Dict[str, Any]]:
        """Rank methods by primary metric."""
        rankings = []
        
        for method, data in results_dict.items():
            score = data['results'].get(primary_metric, 0.0)
            train_samples = data.get('train_samples', 0)
            
            rankings.append({
                'method': method,
                'score': score,
                'train_samples': train_samples,
                'efficiency': score / train_samples if train_samples > 0 else 0
            })
        
        # Sort by primary metric (descending)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return rankings
    
    @staticmethod
    def calculate_efficiency_frontier(results_dict: Dict[str, Dict[str, Any]],
                                    metric: str = 'rougeL_avg') -> List[Dict[str, Any]]:
        """Calculate efficiency frontier (best score per training data size)."""
        frontier_points = []
        
        for method, data in results_dict.items():
            score = data['results'].get(metric, 0.0)
            train_samples = data.get('train_samples', 0)
            
            if train_samples > 0:
                frontier_points.append({
                    'method': method,
                    'score': score,
                    'train_samples': train_samples,
                    'efficiency': score / train_samples
                })
        
        # Sort by training samples
        frontier_points.sort(key=lambda x: x['train_samples'])
        
        return frontier_points
