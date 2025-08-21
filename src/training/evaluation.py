"""
Quick evaluation focused on core metrics for knowledge distillation.
Optimized for fast feedback during experimentation.
"""

import torch
import numpy as np
from datasets import Dataset
from typing import List, Dict, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers/peft not available")
    TRANSFORMERS_AVAILABLE = False

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

# Import our comprehensive metrics calculator
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.metrics import MetricsCalculator
    METRICS_CALCULATOR_AVAILABLE = True
except ImportError:
    logger.warning("Custom metrics calculator not available")
    METRICS_CALCULATOR_AVAILABLE = False

class QuickEvaluator:
    """Fast evaluator using comprehensive metrics (ROUGE + BERTScore)"""
    
    def __init__(self, use_comprehensive_metrics=True):
        self.use_comprehensive_metrics = use_comprehensive_metrics
        self.rouge_scorer = None
        self.metrics_calculator = None
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        # Initialize comprehensive metrics calculator
        if self.use_comprehensive_metrics and METRICS_CALCULATOR_AVAILABLE:
            self.metrics_calculator = MetricsCalculator()
    
    def evaluate_batch(self, predictions, references):
        """Evaluate a batch of predictions using all available metrics"""
        if not predictions or not references:
            return {}
        
        # Use comprehensive metrics if available
        if self.use_comprehensive_metrics and self.metrics_calculator:
            try:
                all_metrics = self.metrics_calculator.calculate_all_metrics(
                    predictions=predictions,
                    references=references
                )
                
                # Format for consistent output
                formatted_metrics = {}
                
                # ROUGE metrics
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    if rouge_type in all_metrics:
                        formatted_metrics[rouge_type] = all_metrics[rouge_type]['fmeasure']
                
                # BERTScore metrics
                if 'bertscore' in all_metrics:
                    bertscore = all_metrics['bertscore']
                    formatted_metrics['bertscore_precision'] = bertscore['precision']
                    formatted_metrics['bertscore_recall'] = bertscore['recall']
                    formatted_metrics['bertscore_f1'] = bertscore['f1']
                
                return formatted_metrics
                
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed: {e}. Falling back to ROUGE-only.")
                
        # Fallback to ROUGE-only evaluation
        return self._evaluate_rouge_only(predictions, references)
    
    def _evaluate_rouge_only(self, predictions, references):
        """Fallback ROUGE-only evaluation"""
        if not self.rouge_scorer:
            logger.warning("No evaluation method available")
            return {}
        
        total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        num_samples = len(predictions)
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in total_scores:
                total_scores[rouge_type] += scores[rouge_type].fmeasure
        
        # Average scores
        return {k: v / num_samples for k, v in total_scores.items()}


class ComprehensiveEvaluator:
    """Comprehensive evaluator with ROUGE + BERTScore metrics for fine-tuned models"""
    
    def __init__(self, base_model_name: str = "meta-llama/Llama-2-7b-hf", use_comprehensive_metrics: bool = True):
        self.base_model_name = base_model_name
        self.use_comprehensive_metrics = use_comprehensive_metrics
        self.rouge_scorer = None
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
            logger.info("✅ ROUGE scorer initialized")
        else:
            logger.warning("❌ ROUGE scorer not available")
    
    def load_model(self, model_path: str) -> Tuple:
        """Load fine-tuned model."""
        logger.info(f"Loading model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        
        # Try to use 8-bit loading if available
        try:
            import bitsandbytes
            model_kwargs["load_in_8bit"] = True
        except ImportError:
            pass
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        return model, tokenizer
    
    def generate_summaries(self, model, tokenizer, test_dataset: Dataset, 
                         max_samples: int = 100) -> Tuple[List[str], List[str]]:
        """Generate summaries for evaluation."""
        logger.info(f"Generating summaries for {min(max_samples, len(test_dataset))} samples")
        
        # Limit samples for quick evaluation
        eval_dataset = test_dataset.shuffle(seed=42).select(range(min(max_samples, len(test_dataset))))
        
        predictions = []
        
        for i, example in enumerate(eval_dataset):
            if i % 10 == 0:
                logger.info(f"Processing sample {i}/{len(eval_dataset)}")
            
            text = example['text']
            prompt = f"Summarize the following bill:\n\n{text}\n\nSummary:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Extract only the generated part
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predictions.append(generated_text.strip())
        
        return predictions, eval_dataset['summary']
    
    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics using all available methods."""
        logger.info("Calculating comprehensive evaluation metrics...")
        
        results = {"num_samples": len(predictions)}
        
        # Try comprehensive metrics first
        if self.use_comprehensive_metrics and METRICS_CALCULATOR_AVAILABLE:
            try:
                # Use comprehensive metrics calculator
                metrics_calc = MetricsCalculator()
                all_metrics = metrics_calc.calculate_all_metrics(
                    predictions=predictions,
                    references=references
                )
                
                # Extract ROUGE metrics
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    if rouge_type in all_metrics:
                        results[f'{rouge_type}_avg'] = all_metrics[rouge_type]['fmeasure']
                        results[f'{rouge_type}_precision'] = all_metrics[rouge_type]['precision']
                        results[f'{rouge_type}_recall'] = all_metrics[rouge_type]['recall']
                
                # Extract BERTScore metrics
                if 'bertscore' in all_metrics:
                    bertscore = all_metrics['bertscore']
                    results['bertscore_precision'] = bertscore['precision']
                    results['bertscore_recall'] = bertscore['recall']
                    results['bertscore_f1'] = bertscore['f1']
                
                logger.info("✅ Comprehensive metrics (ROUGE + BERTScore) calculated successfully")
                return results
                
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed: {e}. Falling back to ROUGE-only.")
        
        # Fallback to ROUGE-only evaluation
        if self.rouge_scorer:
            # Calculate ROUGE scores
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            # Calculate averages
            results.update({
                'rouge1_avg': np.mean(rouge_scores['rouge1']),
                'rouge2_avg': np.mean(rouge_scores['rouge2']),
                'rougeL_avg': np.mean(rouge_scores['rougeL']),
                'rouge1_std': np.std(rouge_scores['rouge1']),
                'rouge2_std': np.std(rouge_scores['rouge2']),
                'rougeL_std': np.std(rouge_scores['rougeL'])
            })
            logger.info("✅ ROUGE metrics calculated")
        else:
            # Fallback to simple metrics
            results.update(self._calculate_simple_metrics(predictions, references))
            logger.info("✅ Simple fallback metrics calculated")
        
        return results
    
    def _calculate_simple_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate simple metrics when ROUGE is not available."""
        length_ratios = []
        word_overlaps = []
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            # Length ratio
            if len(ref) > 0:
                length_ratios.append(len(pred) / len(ref))
            
            # Word overlap (Jaccard similarity)
            if len(ref_words) > 0:
                overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
                word_overlaps.append(overlap)
        
        return {
            'avg_length_ratio': np.mean(length_ratios) if length_ratios else 0.0,
            'avg_word_overlap': np.mean(word_overlaps) if word_overlaps else 0.0,
            'avg_prediction_length': np.mean([len(p.split()) for p in predictions]),
            'avg_reference_length': np.mean([len(r.split()) for r in references])
        }
    
    def quick_evaluate(self, model_path: str, test_dataset: Dataset, 
                      max_samples: int = 100) -> Dict[str, float]:
        """Run complete evaluation pipeline."""
        start_time = datetime.now()
        
        # Load model
        model, tokenizer = self.load_model(model_path)
        
        # Generate predictions
        predictions, references = self.generate_summaries(model, tokenizer, test_dataset, max_samples)
        
        # Calculate metrics
        results = self.evaluate_predictions(predictions, references)
        
        # Add timing info
        eval_time = (datetime.now() - start_time).total_seconds()
        results['evaluation_time_seconds'] = eval_time
        
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        
        return results
