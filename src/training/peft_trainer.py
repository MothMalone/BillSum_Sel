"""
Fast PEFT training with LoRA for knowledge distillation experiments.
Optimized for 30-minute quick iteration and memory efficiency.
"""

import os
import torch
from datasets import Dataset
import logging
import wandb
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import transformers and PEFT
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available")
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    logger.warning("peft not available")
    PEFT_AVAILABLE = False

class QuickPEFTTrainer:
    """Fast PEFT training with LoRA - uses SAME model for selection and training."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):  # Same model for all tasks
        if not TRANSFORMERS_AVAILABLE or not PEFT_AVAILABLE:
            raise ImportError("transformers and peft are required for training")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Realistic settings - prioritize model consistency over speed
        self.quick_mode_settings = {
            "meta-llama/Llama-2-7b-hf": {"max_length": 2048, "batch_size": 2, "max_samples": 1000, "max_steps": 200},
            "meta-llama/Llama-2-13b-hf": {"max_length": 2048, "batch_size": 1, "max_samples": 500, "max_steps": 100},
            "facebook/opt-6.7b": {"max_length": 2048, "batch_size": 2, "max_samples": 800, "max_steps": 150},
            "microsoft/DialoGPT-medium": {"max_length": 1024, "batch_size": 4, "max_samples": 1500, "max_steps": 300},
            "gpt2-xl": {"max_length": 1024, "batch_size": 4, "max_samples": 1200, "max_steps": 250}
        }
    
    def get_optimal_settings(self):
        """Get optimal settings for the selected model - prioritizes consistency over speed."""
        return self.quick_mode_settings.get(
            self.model_name, 
            {"max_length": 2048, "batch_size": 2, "max_samples": 800, "max_steps": 150}  # Conservative defaults
        )
    
    def setup_model(self):
        """Setup model and tokenizer with aggressive memory optimizations."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=os.getenv("HF_TOKEN"),
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Minimal memory model loading
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        # Add authentication if available
        if os.getenv("HF_TOKEN"):
            model_kwargs["use_auth_token"] = os.getenv("HF_TOKEN")
        
        # Try memory optimizations
        try:
            # Try 8-bit loading for larger models
            if "7b" in self.model_name.lower() or "13b" in self.model_name.lower():
                import bitsandbytes
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
                logger.info("Using 8-bit quantization for large model")
        except ImportError:
            logger.warning("bitsandbytes not available, using standard loading")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model with optimizations: {e}")
            # Fallback to basic loading
            fallback_kwargs = {}
            if os.getenv("HF_TOKEN"):
                fallback_kwargs["use_auth_token"] = os.getenv("HF_TOKEN")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **fallback_kwargs
            )
        
        # Minimal LoRA configuration for speed
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=4,  # Very small rank for ultra-fast training
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["c_attn"] if "gpt" in self.model_name.lower() else ["q_proj", "v_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, dataset: Dataset, max_length: int = None) -> Dataset:
        """Prepare dataset for ultra-fast training."""
        settings = self.get_optimal_settings()
        if max_length is None:
            max_length = settings["max_length"]
        
        def format_prompt(example):
            text = example['text']
            summary = example['summary']
            
            # Aggressive truncation for speed
            max_text_words = max_length // 4  # Very short texts
            if len(text.split()) > max_text_words:
                text = ' '.join(text.split()[:max_text_words])
            
            # Simple format for fast processing
            prompt = f"Text: {text}\nSummary: {summary}"
            return {"text": prompt}
        
        formatted_dataset = dataset.map(format_prompt)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False
            )
        
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, output_dir: str, experiment_name: str, 
              num_epochs: int = 1, batch_size: int = None) -> dict:
        """Ultra-fast training for 30-minute target."""
        if self.model is None:
            self.setup_model()
        
        settings = self.get_optimal_settings()
        if batch_size is None:
            batch_size = settings["batch_size"]
        
        # Prepare dataset with aggressive limiting
        train_data = self.prepare_dataset(train_dataset, settings["max_length"])
        
        # Hard limit on samples for speed
        max_samples = settings["max_samples"]
        if len(train_data) > max_samples:
            train_data = train_data.shuffle(seed=42).select(range(max_samples))
            logger.info(f"Limited training to {max_samples} samples for 30-min target")
        
        # Ultra-fast training configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,  # No accumulation for speed
            warmup_steps=10,  # Minimal warmup
            learning_rate=1e-3,  # Higher LR for faster convergence
            logging_steps=50,
            save_steps=9999,  # Disable intermediate saves
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
            run_name=experiment_name,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            max_steps=settings.get("max_steps", 150),  # Reasonable steps for actual convergence
            disable_tqdm=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            data_collator=data_collator,
        )
        
        # Train with timing
        logger.info(f"Starting ultra-fast training ({max_samples} samples, {settings['max_length']} tokens)")
        start_time = datetime.now()
        
        try:
            trainer.train()
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            trainer.save_model()
            
            logger.info(f"Training completed in {training_time:.1f} seconds")
            
            return {
                "model_path": output_dir,
                "training_args": training_args.to_dict(),
                "training_time_seconds": training_time,
                "samples_used": len(train_data),
                "max_length": settings["max_length"],
                "batch_size": batch_size,
                "max_steps": training_args.max_steps
            }
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Training failed after {training_time:.1f}s: {e}")
            return {
                "model_path": output_dir,
                "error": str(e),
                "training_time_seconds": training_time,
                "samples_used": len(train_data) if 'train_data' in locals() else 0
            }
    
    def get_model_size_info(self) -> dict:
        """Get information about model size and parameters."""
        if self.model is None:
            return {"model_name": self.model_name, "loaded": False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "model_name": self.model_name,
            "loaded": True
        }
