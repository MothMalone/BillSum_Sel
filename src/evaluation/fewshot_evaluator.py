#!/usr/bin/env python3
"""
Few-Shot Evaluation Pipeline for BillSum Summarization

Adapted from DialogSum evaluation to work with BillSum dataset and pipeline.
Evaluates base and fine-tuned LLaMA models using few-shot prompting strategy.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# BillSum-specific context and examples
DATASET_CONTEXT = {
    "summary_bill": "You are an expert at summarizing legislative bills. Read the bill text carefully and provide a concise, accurate summary that captures the key provisions and purpose. Keep the summary clear and informative."
}

# Real BillSum examples for few-shot prompting
FEW_SHOT_EXAMPLES = [
    {
        "context": "A bill to require the Secretary of Veterans Affairs to conduct a study on the feasibility of providing dental care to veterans through partnerships with dental schools. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled, SECTION 1. STUDY ON FEASIBILITY OF PROVIDING DENTAL CARE TO VETERANS THROUGH PARTNERSHIPS WITH DENTAL SCHOOLS. (a) Study Required.--The Secretary of Veterans Affairs shall conduct a study on the feasibility of providing dental care to veterans through partnerships between the Department of Veterans Affairs and accredited dental schools.",
        "summary": "Requires the Secretary of Veterans Affairs to study the feasibility of providing dental care to veterans through partnerships with dental schools."
    },
    {
        "context": "A bill to amend title 38, United States Code, to improve the processing of claims for disability compensation by the Department of Veterans Affairs, and for other purposes. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled, SECTION 1. IMPROVEMENTS TO PROCESSING OF CLAIMS FOR DISABILITY COMPENSATION. The Secretary of Veterans Affairs shall implement improvements to the processing of claims for disability compensation to reduce processing times and improve accuracy.",
        "summary": "Amends title 38, United States Code, to improve the processing of disability compensation claims by the Department of Veterans Affairs."
    },
    {
        "context": "A bill to establish a grant program to assist States in establishing or expanding programs to provide services to individuals with autism spectrum disorders. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled, SECTION 1. AUTISM SERVICES GRANT PROGRAM. The Secretary of Health and Human Services may award grants to States to establish or expand programs that provide services to individuals with autism spectrum disorders.",
        "summary": "Establishes a grant program to assist States in providing services to individuals with autism spectrum disorders."
    }
]

def create_few_shot(number_few_shot: int):
    """Create few-shot examples for prompting."""
    shot = []
    for i in range(min(number_few_shot, len(FEW_SHOT_EXAMPLES))):
        shot.append(
            f"Bill: {FEW_SHOT_EXAMPLES[i]['context']}\nSummary: {FEW_SHOT_EXAMPLES[i]['summary']}"
        )
    return shot

def create_request(context=""):
    """Create the request part of the prompt."""
    return f"Bill: {context}\nSummary:"

def create_prompt(task: str = "summary_bill", few_shot: int = 3, context: str = ""):
    """Create the complete few-shot prompt."""
    prompt = DATASET_CONTEXT.get(task, "") + "\n\n"
    request = create_request(context=context)
    if few_shot > 0:
        shot_examples = create_few_shot(few_shot)
        shot_text = '\n\n'.join(shot_examples)
        prompt += f"{shot_text}\n\n{request}"
    else:
        prompt += request
    return prompt

class BillSumFewShotEvaluator:
    """Handles few-shot evaluation of BillSum summarization models."""

    def __init__(self, cache_dir: str = "./cache", log_interval: int = 50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.log_interval = log_interval
        
        # Import metrics here to avoid dependency issues
        try:
            import evaluate
            logger.info("Loading ROUGE metric...")
            self.rouge_scorer = evaluate.load("rouge")
            logger.info("Loading BERTScore metric...")
            self.bertscore_scorer = evaluate.load("bertscore")
            logger.info("Metrics loaded successfully")
        except ImportError as e:
            logger.warning(f"evaluate library not available: {e} - using simple metrics")
            self.rouge_scorer = None
            self.bertscore_scorer = None
        except Exception as e:
            logger.warning(f"Failed to load metrics: {e} - metrics disabled")
            self.rouge_scorer = None
            self.bertscore_scorer = None

    def load_model_and_tokenizer(self, model_path: Optional[str] = None, base_model: str = "meta-llama/Llama-2-7b-hf"):
        """Loads base model or PEFT model from a local path."""
        logger.info(f"Loading base model '{base_model}' with optimizations...")
        
        # Load base model with memory optimizations
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        if model_path:
            logger.info(f"Loading LoRA adapter from '{model_path}'...")
            model = PeftModel.from_pretrained(base, model_path)
        else:
            logger.info("Using base model without fine-tuning...")
            model = base
            
        model.eval()
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    def generate_summaries(self, model, tokenizer, dataset: Dataset, batch_size: int = 2, 
                          max_new_tokens: int = 128, num_few_shots: int = 3, 
                          output_dir: Optional[Path] = None, max_samples: Optional[int] = None):
        """Generate summaries with incremental logging."""
        
        # Limit dataset size if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            
        logger.info(f"Starting summary generation with {num_few_shots} few-shot examples on {len(dataset)} samples...")
        predictions, references = [], []
        
        # Create results tracking
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
            incremental_results_file = output_dir / "incremental_results.csv"
            results_df = pd.DataFrame(columns=["num_samples", "rouge1", "rouge2", "rougeL", "rougeLsum", 
                                              "bertscore_precision", "bertscore_recall", "bertscore_f1"])
        
        total_processed = 0
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Summaries"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            # Get texts and summaries
            texts = batch['text']
            summaries = batch['summary']
            
            # Create prompts using few-shot strategy
            prompts = [create_prompt(context=text[:2000], few_shot=num_few_shots) for text in texts]  # Limit context length
            
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens, 
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,  # Enable sampling for diversity
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                # Decode only the newly generated tokens
                input_lengths = inputs['input_ids'].shape[1]
                newly_generated_tokens = outputs[:, input_lengths:]
                decoded_preds = tokenizer.batch_decode(newly_generated_tokens, skip_special_tokens=True)
                
                cleaned_preds = [pred.strip() for pred in decoded_preds]
                predictions.extend(cleaned_preds)
                references.extend(summaries)
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Add empty predictions for failed batch - make sure lengths match
                batch_size_actual = len(texts)
                predictions.extend([""] * batch_size_actual)
                references.extend(summaries)
                
            total_processed += len(batch)
            
            # Log progress periodically
            if total_processed % self.log_interval == 0:
                logger.info(f"Processed {total_processed}/{len(dataset)} samples...")
                
                if output_dir and self.rouge_scorer is not None:
                    # Compute intermediate metrics
                    metrics = self.compute_metrics(predictions, references)
                    
                    # Save incremental results
                    new_row = {
                        "num_samples": total_processed,
                        "rouge1": metrics["rouge1"],
                        "rouge2": metrics["rouge2"],
                        "rougeL": metrics["rougeL"],
                        "rougeLsum": metrics["rougeLsum"],
                        "bertscore_precision": metrics.get("bertscore_precision", 0.0),
                        "bertscore_recall": metrics.get("bertscore_recall", 0.0),
                        "bertscore_f1": metrics.get("bertscore_f1", 0.0)
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    results_df.to_csv(incremental_results_file, index=False)
                    
                    logger.info(f"Intermediate ROUGE-L: {metrics['rougeL']:.4f}")

        logger.info(f"Generated {len(predictions)} summaries.")
        return predictions, references

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics between predictions and references."""
        results = {}
        
        logger.info(f"Computing metrics for {len(predictions)} predictions and {len(references)} references")
        
        # Ensure we have non-empty predictions and references
        if not predictions or not references or len(predictions) != len(references):
            logger.warning(f"Mismatch in predictions ({len(predictions)}) and references ({len(references)})")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0, 
                   'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
        
        # Filter out empty predictions
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        if not filtered_pairs:
            logger.warning("No valid prediction-reference pairs found")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0, 
                   'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
        
        filtered_predictions, filtered_references = zip(*filtered_pairs)
        logger.info(f"Using {len(filtered_pairs)} valid pairs for metric computation")
        
        if self.rouge_scorer is not None:
            try:
                logger.info("Attempting ROUGE computation...")
                rouge_scores = self.rouge_scorer.compute(
                    predictions=list(filtered_predictions), 
                    references=list(filtered_references), 
                    use_stemmer=True
                )
                logger.info(f"ROUGE scores raw: {rouge_scores}")
                results.update({
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rougeL': rouge_scores['rougeL'],
                    'rougeLsum': rouge_scores['rougeLsum'],
                })
                logger.info(f"ROUGE computed successfully: ROUGE-L={rouge_scores['rougeL']:.4f}")
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")
                import traceback
                traceback.print_exc()
                results.update({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0})
        else:
            logger.warning("ROUGE scorer not available")
            results.update({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0})
        
        if self.bertscore_scorer is not None:
            try:
                logger.info("Attempting BERTScore computation...")
                bert_scores = self.bertscore_scorer.compute(
                    predictions=list(filtered_predictions), 
                    references=list(filtered_references), 
                    lang="en"
                )
                logger.info(f"BERTScore computed, F1 mean: {np.mean(bert_scores['f1']):.4f}")
                results.update({
                    'bertscore_precision': np.mean(bert_scores['precision']),
                    'bertscore_recall': np.mean(bert_scores['recall']),
                    'bertscore_f1': np.mean(bert_scores['f1']),
                })
                logger.info(f"BERTScore computed successfully: F1={np.mean(bert_scores['f1']):.4f}")
            except Exception as e:
                logger.warning(f"BERTScore computation failed: {e}")
                import traceback
                traceback.print_exc()
                results.update({'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0})
        else:
            logger.warning("BERTScore scorer not available")
            results.update({'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0})
        
        return results

    def evaluate_model(self, model_path: Optional[str], base_model: str, eval_dataset: Dataset,
                      output_dir: str, num_few_shots: int = 3, batch_size: int = 2, 
                      max_samples: Optional[int] = None):
        """Complete evaluation pipeline."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load model
        model, tokenizer = self.load_model_and_tokenizer(model_path, base_model)
        
        # Generate summaries
        predictions, references = self.generate_summaries(
            model, tokenizer, eval_dataset,
            batch_size=batch_size,
            num_few_shots=num_few_shots,
            output_dir=output_path,
            max_samples=max_samples
        )
        
        # Compute final metrics
        logger.info("Computing final metrics...")
        scores = self.compute_metrics(predictions, references)
        logger.info(f"Final scores computed: {scores}")
        
        # Save results
        model_key = Path(model_path).name if model_path else "base_model"
        results_file = output_path / f"results_{model_key}.json"
        with open(results_file, 'w') as f:
            json.dump(scores, f, indent=4)
        logger.info(f"Results saved to {results_file}")
        
        # Save predictions - make sure all arrays have the same length
        actual_dataset = eval_dataset.select(range(min(len(predictions), len(eval_dataset))))
        df = pd.DataFrame({
            "text": [d['text'][:500] + "..." for d in actual_dataset],  # Truncate for readability
            "reference": references[:len(actual_dataset)], 
            "prediction": predictions[:len(actual_dataset)]
        })
        df.to_csv(output_path / f"predictions_{model_key}.csv", index=False)
        
        logger.info(f"Evaluation complete. Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print(f"FEW-SHOT EVALUATION SUMMARY FOR: {model_key}")
        print("="*80)
        print(f"{'Metric':<20} {'Score'}")
        print("-"*80)
        for key, value in scores.items():
            print(f"{key:<20} {value:.4f}")
        print("="*80)
        
        return scores
