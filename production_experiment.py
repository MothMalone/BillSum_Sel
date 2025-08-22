#!/usr/bin/env python3
"""
Fixed Production BillSum Knowledge Distillation Experiment for Vast.ai
Resolves gradient computation issues in PEFT training.
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
    """PEFT trainer optimized for 16GB GPU memory constraints with gradient fix."""
    
    def __init__(self):
        self.model_name = "meta-llama/Llama-2-7b-hf"  # LLaMA-2-7B model
        self.model = None  # Initialize model attribute
        self.tokenizer = None  # Initialize tokenizer attribute
        self.config = {
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "max_length": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "warmup_steps": 10,
            "bf16": True  # Enable bf16 for memory efficiency
        }
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with memory optimization and gradient fix."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                BitsAndBytesConfig
            )
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # CRITICAL FIX: Prepare model for k-bit training
            # This enables gradients for quantized parameters
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=self.config["lora_dropout"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                inference_mode=False,  # Ensure we're in training mode
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            # CRITICAL FIX: Explicitly enable training mode and gradients
            self.model.train()
            
            # Verify gradient setup
            grad_enabled_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad_enabled_params += 1
            
            logger.info(f"Parameters with gradients enabled: {grad_enabled_params}")
            
            # Enable gradients for LoRA parameters explicitly
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad_(True)
            
            logger.info("✅ Model and tokenizer loaded successfully with gradient fix")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def prepare_dataset(self, dataset, max_samples: int = 100):
        """Prepare dataset for training with proper tensor conversion."""
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
            target = f"{summary}{self.tokenizer.eos_token}"
            
            # Tokenize the full text (prompt + target)
            full_text = prompt + target
            
            model_inputs = self.tokenizer(
                full_text,
                max_length=self.config["max_length"],
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # Create labels - we want to predict the summary part only
            prompt_inputs = self.tokenizer(
                prompt,
                max_length=self.config["max_length"] - 100,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            input_ids = model_inputs["input_ids"]
            labels = input_ids.copy()
            
            # Mask the prompt tokens (set to -100 so they're ignored in loss)
            prompt_length = len(prompt_inputs["input_ids"])
            labels[:prompt_length] = [-100] * prompt_length
            
            # CRITICAL FIX: Ensure all outputs are lists (not tensors)
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": model_inputs.get("attention_mask", [1] * len(input_ids))
            }
        
        formatted_dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        logger.info(f"✅ Dataset prepared: {len(formatted_dataset)} samples")
        return formatted_dataset
    
    def train(self, train_dataset, output_dir: str, method_name: str):
        """Train the model with LoRA and gradient fixes."""
        logger.info(f"🚀 Starting PEFT training for {method_name}")
        
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
            
            # Force fresh model setup for each training run
            if hasattr(self, 'model'):
                del self.model
                del self.tokenizer
                clear_gpu_memory()
            
            # Setup fresh model for this training
            if not self.setup_model_and_tokenizer():
                raise Exception("Failed to setup model")
            
            # Prepare dataset
            formatted_dataset = self.prepare_dataset(train_dataset)
            
            # CRITICAL FIX: Disable gradient checkpointing for quantized models
            # It conflicts with 4-bit training
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                learning_rate=self.config["learning_rate"],
                num_train_epochs=self.config["num_epochs"],
                logging_steps=5,
                save_steps=500,
                save_total_limit=1,
                remove_unused_columns=True,
                dataloader_pin_memory=False,
                bf16=self.config["bf16"],
                gradient_checkpointing=False,  # DISABLED for 4-bit training
                report_to=None,  # Disable wandb/tensorboard
                run_name=f"{method_name}_peft_training",
                # Additional fixes
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                warmup_steps=self.config["warmup_steps"],
                weight_decay=0.01,
                lr_scheduler_type="linear"
            )
            
            # Data collator with proper padding
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
                return_tensors="pt"  # Ensure tensor output
            )
            
            # CRITICAL FIX: Custom trainer class to handle gradient issues
            class FixedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    """Override to ensure proper gradient computation."""
                    # Ensure model is in training mode
                    model.train()
                    
                    # Move inputs to device if needed
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = model(**inputs)
                    loss = outputs.get("loss")
                    
                    if loss is None:
                        # Manually compute loss if not returned
                        logits = outputs.get("logits")
                        labels = inputs.get("labels")
                        
                        if logits is not None and labels is not None:
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            
                            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                    
                    return (loss, outputs) if return_outputs else loss
            
            # Use fixed trainer
            trainer = FixedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=formatted_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Final verification before training
            self.model.train()
            
            # Debug: Check gradient setup one more time
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"📊 Final check - Trainable parameters: {trainable_params:,} / {total_params:,}")
            
            if trainable_params == 0:
                logger.error("❌ No trainable parameters found!")
                return {
                    "success": False,
                    "error": "No trainable parameters",
                    "output_dir": output_dir,
                    "dataset_size": len(formatted_dataset)
                }
            
            # Verify a sample batch can be processed
            try:
                sample_batch = data_collator([formatted_dataset[0]])
                sample_batch = {k: v.to(self.model.device) for k, v in sample_batch.items()}
                
                # Test forward pass
                self.model.train()
                with torch.enable_grad():
                    test_outputs = self.model(**sample_batch)
                    test_loss = test_outputs.loss
                    logger.info(f"✅ Sample forward pass successful, loss: {test_loss.item():.4f}")
                
                # Test backward pass
                test_loss.backward()
                logger.info("✅ Sample backward pass successful")
                
                # Clear test gradients
                self.model.zero_grad()
                
            except Exception as e:
                logger.error(f"❌ Sample batch test failed: {e}")
                return {
                    "success": False,
                    "error": f"Sample batch test failed: {str(e)}",
                    "output_dir": output_dir,
                    "dataset_size": len(formatted_dataset)
                }
            
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
                "model": self.model_name,
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

# [Rest of the code remains the same - ProductionEvaluator, selection functions, main experiment]

class ProductionEvaluator:
    """Production evaluator with real model inference."""
    
    def __init__(self):
        self.model_name = "meta-llama/Llama-2-7b-hf"  # LLaMA-2-7B model
        self.model = None  # Initialize model attribute
        self.tokenizer = None  # Initialize tokenizer attribute
    
    def setup_model_and_tokenizer(self):
        """Setup base model and tokenizer for evaluation."""
        logger.info(f"Loading base model: {self.model_name}")
        
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                BitsAndBytesConfig
            )
            
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            logger.info("✅ Base model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load base model: {e}")
            return False
    
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
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
            # Truncate input text to reasonable length
            text_words = text.split()[:200]  # Limit to ~200 words
            text = ' '.join(text_words)
            
            # Use a clearer prompt format
            prompt = f"Summarize the following text in 1-2 sentences:\n\n{text}\n\nSummary:"
            
            inputs = self.tokenizer(
                prompt,
                max_length=512,  # Shorter input to leave room for generation
                truncation=True,
                padding=False,
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,  # Ensure we generate tokens
                    min_new_tokens=10,   # Force minimum generation
                    do_sample=True,      # Enable sampling
                    temperature=0.7,     # Lower temperature for more focused output
                    top_p=0.9,          # Nucleus sampling
                    top_k=50,           # Top-k sampling
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Avoid repetition
                    no_repeat_ngram_size=3   # Avoid repeated phrases
                )
            
            # Decode only the new tokens (skip the input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]  # Only new tokens
            
            if len(generated_tokens) == 0:
                return ""
                
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the output
            summary = generated_text.strip()
            
            # Remove any remaining prompt artifacts
            if summary.lower().startswith('summary:'):
                summary = summary[8:].strip()
                
            # Take first paragraph if multi-paragraph
            if '\n\n' in summary:
                summary = summary.split('\n\n')[0].strip()
                
            # Limit to reasonable length (first 2-3 sentences)
            sentences = summary.split('.')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]).strip()
                if not summary.endswith('.'):
                    summary += '.'
            
            return summary if summary else ""
            
        except Exception as e:
            logger.error(f"❌ Failed to generate summary: {e}")
            return ""
    
    def evaluate_base_model(self, test_dataset, max_samples: int = 20):
        """Evaluate base model without fine-tuning."""
        logger.info(f"📊 Evaluating base model (no fine-tuning)")
        
        # Setup base model and tokenizer
        if not self.setup_model_and_tokenizer():
            return {"error": "Failed to setup base model"}
        
        # Sample test data
        eval_dataset = test_dataset.shuffle(seed=42).select(range(min(max_samples, len(test_dataset))))
        
        predictions = []
        references = []
        
        logger.info(f"Generating base model predictions for {len(eval_dataset)} samples...")
        
        for i, example in enumerate(eval_dataset):
            if i % 5 == 0:
                logger.info(f"Processing sample {i+1}/{len(eval_dataset)}")
            
            text = example.get('text', '').strip()
            reference = example.get('summary', '').strip()
            
            if not text or not reference:
                continue
            
            # Generate prediction with base model
            prediction = self.generate_summary(text)
            
            logger.info(f"   📝 Generated prediction length: {len(prediction) if prediction else 0}")
            
            if prediction and reference:
                predictions.append(prediction)
                references.append(reference)
            else:
                logger.warning(f"   ⚠️ Skipping sample - prediction: {bool(prediction)}, reference: {bool(reference)}")
        
        logger.info(f"✅ Generated {len(predictions)} base model predictions")
        
        # Check if we have any predictions
        if len(predictions) == 0:
            logger.error("❌ No valid predictions generated for base model")
            return {
                "error": "No valid predictions generated",
                "bertscore_f1": 0.0,
                "rouge_scores": {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "rougeLsum": 0.0
                },
                "num_predictions": 0
            }
        
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
            
            logger.info(f"📈 Base Model ROUGE-Lsum: {results['rougeLsum']:.4f}")
            logger.info(f"📈 Base Model BERTScore F1: {results['bertscore_f1']:.4f}")
            
            # Clear memory
            del self.model, self.tokenizer
            clear_gpu_memory()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Base model evaluation failed: {e}")
            # Clear memory on error
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            clear_gpu_memory()
            return {"error": str(e)}
    
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
    
    # Base model evaluation (baseline)
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: BASE MODEL EVALUATION (BASELINE)")
    logger.info(f"{'='*60}")
    
    try:
        start_time = time.time()
        base_eval_result = evaluator.evaluate_base_model(test_dataset, n_eval_samples)
        base_eval_time = time.time() - start_time
        
        if base_eval_result and "bertscore_f1" in base_eval_result:
            results["base_model"] = {
                "method": "base_model",
                "selection_time": 0.0,  # No selection needed
                "training_time": 0.0,   # No training
                "dataset_size": 0,      # No training data
                "evaluation": base_eval_result,
                "evaluation_time": base_eval_time
            }
            
            bertscore_f1 = base_eval_result.get('bertscore_f1', 0.0)
            rougeLsum = base_eval_result.get('rougeLsum', base_eval_result.get('rougeL', 0.0))
            logger.info(f"✅ Base Model - BERTScore F1: {bertscore_f1:.4f} | ROUGE-Lsum: {rougeLsum:.4f}")
        else:
            logger.error("❌ Base model evaluation failed")
            results["base_model"] = {
                "method": "base_model",
                "error": "Base model evaluation failed"
            }
    except Exception as e:
        logger.error(f"❌ Base model evaluation error: {e}")
        results["base_model"] = {
            "method": "base_model", 
            "error": str(e)
        }
    
    # Run fine-tuning experiments
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
            
            # Handle base model vs fine-tuned models
            if method_name == "base_model":
                print(f"  Type: BASELINE (No fine-tuning)")
                if "evaluation_time" in result:
                    print(f"  Evaluation Time: {result['evaluation_time']:.2f}s")
            else:
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
