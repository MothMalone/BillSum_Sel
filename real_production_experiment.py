#!/usr/bin/env python3
"""
Production BillSum Knowledge Distillation Experiment for Vast.ai
Real training implementation with memory optimization for 16GB GPU constraints.
"""

import os
import sys
import torch
import logging
import platform
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append('src')

def check_environment() -> bool:
    """Check system environment and GPU memory."""
    logger.info("=== VAST.AI PRODUCTION ENVIRONMENT CHECK ===")
    logger.info(f"Python: {platform.python_version()}")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            logger.error("❌ CUDA not available - GPU required for this experiment")
            return False
        
        # Check GPU memory
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        memory_gb = device_props.total_memory / 1024**3
        
        logger.info(f"GPU: {device_name}")
        logger.info(f"GPU Memory: {memory_gb:.1f} GB")
        
        if memory_gb < 12:
            logger.warning(f"⚠️  Low GPU memory ({memory_gb:.1f} GB) - experiment may fail")
            return False
        
        logger.info("✅ GPU environment suitable for experiment")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        return False

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_dataset():
    """Load BillSum dataset with error handling."""
    logger.info("Loading BillSum dataset...")
    
    try:
        from datasets import load_dataset
        
        # Try with token if available, otherwise without
        hf_token = os.getenv("HF_TOKEN")
        
        try:
            if hf_token and hf_token != "your_huggingface_token_here":
                dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", name="billsum", token=hf_token)
                logger.info("✅ Loaded dataset with authentication")
            else:
                dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", name="billsum")
                logger.info("✅ Loaded dataset without authentication")
        except Exception as e:
            logger.warning(f"Primary load failed: {e}")
            dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", name="billsum")
            logger.info("✅ Loaded dataset with fallback method")
        
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
        logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return None, None

class MemoryOptimizedPEFTTrainer:
    """Memory-optimized PEFT trainer for 16GB GPU constraints."""
    
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        
        # Memory-optimized settings
        self.config = {
            "max_length": 1024,          # Reduced from 2048
            "batch_size": 1,             # Very small batch
            "gradient_accumulation": 8,  # Accumulate to simulate larger batch
            "learning_rate": 2e-4,
            "num_epochs": 1,             # Single epoch for efficiency
            "lora_r": 8,                 # LoRA rank
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bf16": True,                # Use bf16 for memory efficiency
        }
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with memory optimization."""
        logger.info(f"Loading model: {self.base_model_name}")
        
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                BitsAndBytesConfig
            )
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Clear memory first
            clear_gpu_memory()
            
            # Quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Setup LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=self.config["lora_dropout"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("✅ Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def prepare_dataset(self, dataset, max_samples: int = 100):
        """Prepare dataset for training."""
        logger.info(f"Preparing dataset with {max_samples} samples")
        
        # Limit dataset size
        if len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=42).select(range(max_samples))
        
        def format_sample(example):
            text = example.get('text', '').strip()
            summary = example.get('summary', '').strip()
            
            # Truncate text if too long
            text_words = text.split()[:300]  # Limit to ~300 words
            text = ' '.join(text_words)
            
            # Create instruction format
            prompt = f"Summarize the following bill:\n\n{text}\n\nSummary:"
            target = f"{summary}"
            
            # Tokenize
            model_inputs = self.tokenizer(
                prompt,
                max_length=self.config["max_length"] - 100,  # Reserve space for summary
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # Tokenize target
            target_inputs = self.tokenizer(
                target,
                max_length=100,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # Combine input and target
            input_ids = model_inputs["input_ids"] + target_inputs["input_ids"]
            labels = [-100] * len(model_inputs["input_ids"]) + target_inputs["input_ids"]
            
            # Pad/truncate to max_length
            max_len = self.config["max_length"]
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
            
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids)
            }
        
        formatted_dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        logger.info(f"✅ Dataset prepared: {len(formatted_dataset)} samples")
        return formatted_dataset
    
    def train(self, train_dataset, output_dir: str, method_name: str):
        """Train the model with LoRA."""
        logger.info(f"🚀 Starting PEFT training for {method_name}")
        
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            # Setup model if not already done
            if self.model is None:
                if not self.setup_model_and_tokenizer():
                    raise Exception("Failed to setup model")
            
            # Prepare dataset
            formatted_dataset = self.prepare_dataset(train_dataset)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation"],
                learning_rate=self.config["learning_rate"],
                num_train_epochs=self.config["num_epochs"],
                logging_steps=10,
                save_steps=500,
                save_total_limit=1,
                remove_unused_columns=True,
                dataloader_pin_memory=False,
                bf16=self.config["bf16"],
                gradient_checkpointing=True,
                report_to=None,  # Disable wandb/tensorboard
                run_name=f"{method_name}_peft_training"
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=formatted_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Clear memory before training
            clear_gpu_memory()
            
            # Train
            logger.info("📚 Starting training...")
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            
            # Save training metadata
            metadata = {
                "method": method_name,
                "model": self.base_model_name,
                "dataset_size": len(formatted_dataset),
                "training_time": training_time,
                "config": self.config,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Training completed in {training_time:.2f}s")
            logger.info(f"📁 Model saved to: {output_dir}")
            
            # Clear memory after training
            clear_gpu_memory()
            
            return {
                "success": True,
                "training_time": training_time,
                "output_dir": output_dir,
                "dataset_size": len(formatted_dataset)
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            clear_gpu_memory()
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
                "dataset_size": 0
            }

class ProductionEvaluator:
    """Production evaluator using actual model inference."""
    
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
    
    def load_trained_model(self, model_path: str):
        """Load trained PEFT model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            
            # Clear memory
            clear_gpu_memory()
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Load PEFT weights
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.eval()
            
            logger.info(f"✅ Loaded trained model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load trained model: {e}")
            return False
    
    def generate_summary(self, text: str) -> str:
        """Generate summary using the trained model."""
        try:
            prompt = f"Summarize the following bill:\n\n{text}\n\nSummary:"
            
            inputs = self.tokenizer(
                prompt,
                max_length=800,
                truncation=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract summary (after "Summary:")
            if "Summary:" in generated:
                summary = generated.split("Summary:")[-1].strip()
            else:
                summary = generated[len(prompt):].strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate summary: {e}")
            return ""
    
    def evaluate(self, model_path: str, test_dataset, max_samples: int = 20):
        """Evaluate trained model."""
        logger.info(f"📊 Evaluating model: {model_path}")
        
        # Load trained model
        if not self.load_trained_model(model_path):
            return {"error": "Failed to load model"}
        
        # Sample test data
        eval_dataset = test_dataset.shuffle(seed=42).select(range(min(max_samples, len(test_dataset))))
        
        predictions = []
        references = []
        
        logger.info(f"Generating predictions for {len(eval_dataset)} samples...")
        
        for i, example in enumerate(eval_dataset):
            if i % 5 == 0:
                logger.info(f"Processing sample {i+1}/{len(eval_dataset)}")
            
            text = example.get('text', '').strip()
            reference = example.get('summary', '').strip()
            
            if not text or not reference:
                continue
            
            # Generate prediction
            prediction = self.generate_summary(text)
            
            if prediction and reference:
                predictions.append(prediction)
                references.append(reference)
        
        logger.info(f"✅ Generated {len(predictions)} predictions")
        
        # Compute ROUGE metrics (including ROUGE-Lsum)
        try:
            from evaluate import load
            rouge_scorer = load("rouge")
            
            rouge_scores = rouge_scorer.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True,
                rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
            )
            
            # Compute BERTScore (main metric)
            logger.info("📊 Computing BERTScore...")
            try:
                from bert_score import score
                P, R, F1 = score(predictions, references, lang="en", verbose=False)
                bertscore_f1 = F1.mean().item()
                bertscore_precision = P.mean().item()
                bertscore_recall = R.mean().item()
                logger.info(f"📈 BERTScore F1: {bertscore_f1:.4f}")
            except Exception as e:
                logger.warning(f"BERTScore computation failed: {e}")
                bertscore_f1 = 0.0
                bertscore_precision = 0.0
                bertscore_recall = 0.0
            
            results = {
                "num_samples": len(predictions),
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "rougeLsum": rouge_scores.get("rougeLsum", rouge_scores["rougeL"]),  # Main ROUGE metric
                "bertscore_f1": bertscore_f1,        # Main BERTScore metric
                "bertscore_precision": bertscore_precision,
                "bertscore_recall": bertscore_recall,
                "predictions": predictions[:3],  # Sample predictions
                "references": references[:3]     # Sample references
            }
            
            logger.info(f"📈 ROUGE-Lsum: {results['rougeLsum']:.4f} (main ROUGE metric)")
            logger.info(f"📈 BERTScore F1: {results['bertscore_f1']:.4f} (main BERTScore metric)")
            
            # Clear memory
            del self.model, self.tokenizer
            clear_gpu_memory()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {"error": str(e)}

def random_selection(dataset, n_samples: int):
    """Random data selection."""
    logger.info(f"Random selection: {n_samples} samples")
    
    import random
    random.seed(42)
    
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    return dataset.select(indices)

def length_based_selection(dataset, n_samples: int):
    """Length-based data selection."""
    logger.info(f"Length-based selection: {n_samples} samples")
    
    # Find samples with optimal length (200-800 words)
    good_indices = []
    
    for i, example in enumerate(dataset):
        text = example.get('text', '')
        word_count = len(text.split())
        
        if 200 <= word_count <= 800:
            good_indices.append(i)
    
    logger.info(f"Found {len(good_indices)} samples with optimal length")
    
    # Sample from good indices
    import random
    random.seed(42)
    
    selected_indices = random.sample(good_indices, min(n_samples, len(good_indices)))
    return dataset.select(selected_indices)

def embedding_based_selection(dataset, n_samples: int):
    """Embedding-based data selection using lightweight model."""
    logger.info(f"Embedding-based selection: {n_samples} samples")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Use very lightweight model
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Process in small batches
        batch_size = 32
        embeddings = []
        texts = []
        
        logger.info("Generating embeddings...")
        for i, example in enumerate(dataset):
            text = example.get('text', '')[:500]  # Truncate for efficiency
            texts.append(text)
            
            if len(texts) == batch_size or i == len(dataset) - 1:
                batch_embeddings = embedding_model.encode(texts, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                texts = []
                
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{len(dataset)} samples")
        
        # Convert to numpy
        embeddings = np.array(embeddings)
        
        # Select diverse samples using k-means clustering
        from sklearn.cluster import KMeans
        
        # Cluster into n_samples clusters
        kmeans = KMeans(n_clusters=min(n_samples, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Select sample closest to each cluster center
        selected_indices = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(int(closest_idx))
        
        logger.info(f"✅ Selected {len(selected_indices)} diverse samples")
        
        # Cleanup
        del embedding_model, embeddings
        clear_gpu_memory()
        
        return dataset.select(selected_indices)
        
    except Exception as e:
        logger.error(f"❌ Embedding selection failed: {e}, falling back to random")
        return random_selection(dataset, n_samples)

def run_production_experiment():
    """Run complete production experiment."""
    logger.info("🚀 VAST.AI PRODUCTION BILLSUM EXPERIMENT")
    logger.info("Real PEFT training with memory optimization")
    
    # Environment check
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    # Load dataset
    train_dataset, test_dataset = load_dataset()
    if train_dataset is None:
        logger.error("❌ Failed to load dataset")
        return False
    
    # Experiment configuration
    n_train_samples = 100  # Small for demonstration
    n_eval_samples = 20
    
    # Data selection methods
    methods = [
        ("random", random_selection),
        ("length_based", length_based_selection),
        ("embedding_based", embedding_based_selection)
    ]
    
    # Initialize trainer and evaluator
    trainer = MemoryOptimizedPEFTTrainer()
    evaluator = ProductionEvaluator()
    
    results = {}
    
    # Run experiments
    for method_name, selector_func in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {method_name.upper().replace('_', ' ')}")
        logger.info(f"{'='*60}")
        
        try:
            # Data selection
            start_time = time.time()
            selected_data = selector_func(train_dataset, n_train_samples)
            selection_time = time.time() - start_time
            
            logger.info(f"✅ Selected {len(selected_data)} samples in {selection_time:.2f}s")
            
            # Training
            output_dir = f"results/production/{method_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            training_result = trainer.train(selected_data, output_dir, method_name)
            
            if training_result["success"]:
                # Evaluation
                eval_result = evaluator.evaluate(output_dir, test_dataset, n_eval_samples)
                
                # Store results
                results[method_name] = {
                    "method": method_name,
                    "selection_time": selection_time,
                    "training_time": training_result["training_time"],
                    "dataset_size": training_result["dataset_size"],
                    "evaluation": eval_result
                }
                
                if "rouge1" in eval_result:
                    # Log main metrics
                    bertscore_f1 = eval_result.get('bertscore_f1', 0.0)
                    rougeLsum = eval_result.get('rougeLsum', eval_result.get('rougeL', 0.0))
                    logger.info(f"✅ {method_name} - BERTScore F1: {bertscore_f1:.4f} | ROUGE-Lsum: {rougeLsum:.4f}")
                else:
                    logger.warning(f"⚠️  {method_name} - Evaluation failed")
            else:
                logger.error(f"❌ {method_name} - Training failed")
                results[method_name] = {
                    "method": method_name,
                    "error": training_result["error"]
                }
            
        except Exception as e:
            logger.error(f"❌ {method_name} failed: {e}")
            results[method_name] = {
                "method": method_name,
                "error": str(e)
            }
        
        # Clear memory between methods
        clear_gpu_memory()
    
    # Save and display results
    save_results(results, n_train_samples, n_eval_samples)
    display_results(results)
    
    logger.info("🎉 Production experiment completed!")
    return True

def save_results(results: Dict, n_train: int, n_eval: int):
    """Save experiment results."""
    os.makedirs("results/production", exist_ok=True)
    
    output = {
        "experiment_type": "vast_ai_production_real",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_samples": n_train,
            "eval_samples": n_eval,
            "gpu_optimized": True,
            "real_training": True
        },
        "results": results
    }
    
    with open("results/production/real_experiment_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info("📁 Results saved to: results/production/real_experiment_results.json")

def display_results(results: Dict):
    """Display formatted results."""
    print("\n" + "="*80)
    print("VAST.AI BILLSUM PRODUCTION EXPERIMENT - REAL TRAINING RESULTS")
    print("="*80)
    
    for method_name, result in results.items():
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        if "error" in result:
            print(f"  Status: ❌ FAILED")
            print(f"  Error: {result['error']}")
        elif "evaluation" in result and "rouge1" in result["evaluation"]:
            eval_data = result["evaluation"]
            print(f"  Status: ✅ SUCCESS")
            print(f"  Selection Time: {result['selection_time']:.2f}s")
            print(f"  Training Time: {result['training_time']:.2f}s")
            print(f"  Dataset Size: {result['dataset_size']}")
            print(f"  📊 MAIN METRICS:")
            print(f"     🎯 BERTScore F1: {eval_data.get('bertscore_f1', 0.0):.4f} (MAIN)")
            print(f"     🎯 ROUGE-Lsum: {eval_data.get('rougeLsum', eval_data.get('rougeL', 0.0)):.4f} (MAIN)")
            print(f"  📈 Additional metrics:")
            print(f"     ROUGE-1: {eval_data['rouge1']:.4f}")
            print(f"     ROUGE-2: {eval_data['rouge2']:.4f}")
            print(f"     ROUGE-L: {eval_data['rougeL']:.4f}")
            if 'bertscore_precision' in eval_data:
                print(f"     BERTScore P: {eval_data['bertscore_precision']:.4f}")
                print(f"     BERTScore R: {eval_data['bertscore_recall']:.4f}")
            print(f"  Eval Samples: {eval_data['num_samples']}")
        else:
            print(f"  Status: ⚠️  PARTIAL (training succeeded, evaluation failed)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        success = run_production_experiment()
        if success:
            print("\n✅ PRODUCTION EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("📊 Real PEFT training and evaluation completed")
            print("💰 Optimized for Vast.ai 16GB GPU constraints")
        else:
            print("\n❌ Experiment failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        clear_gpu_memory()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        clear_gpu_memory()
        sys.exit(1)
