#!/usr/bin/env python3
"""
Quick functionality test for BillSum experiment
Tests the core components without running full training
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu():
    """Test GPU availability and memory"""
    logger.info("🔧 Testing GPU environment...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info(f"✅ GPU: {gpu_name}")
    logger.info(f"✅ Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 15:
        logger.warning(f"⚠️  Low VRAM: {gpu_memory:.1f}GB (need 15GB+)")
    
    return True

def test_dataset_loading():
    """Test BillSum dataset loading"""
    logger.info("📊 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("billsum", split="test[:3]")
        logger.info(f"✅ Loaded {len(dataset)} samples from BillSum")
        
        # Show sample
        sample = dataset[0]
        logger.info(f"✅ Sample text length: {len(sample['text'])} chars")
        logger.info(f"✅ Sample summary length: {len(sample['summary'])} chars")
        
        return True
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False

def test_model_loading():
    """Test model and tokenizer loading"""
    logger.info("🤖 Testing model loading...")
    
    try:
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded")
        
        # Test model loading with quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        logger.info("🔧 Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logger.info("✅ Model loaded with quantization")
        
        # Test memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✅ GPU memory used: {memory_used:.2f} GB")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics"""
    logger.info("📈 Testing evaluation metrics...")
    
    try:
        # Test ROUGE
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        test_pred = "This is a test prediction summary."
        test_ref = "This is a reference summary for testing."
        
        scores = scorer.score(test_ref, test_pred)
        logger.info(f"✅ ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        
        # Test BERTScore
        from bert_score import score
        P, R, F1 = score([test_pred], [test_ref], lang="en", verbose=False)
        logger.info(f"✅ BERTScore F1: {F1.item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Evaluation metrics failed: {e}")
        return False

def test_peft_setup():
    """Test PEFT configuration"""
    logger.info("🔧 Testing PEFT setup...")
    
    try:
        from peft import LoraConfig, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        logger.info("✅ LoRA configuration created")
        
        return True
    except Exception as e:
        logger.error(f"❌ PEFT setup failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting BillSum Quick Functionality Test")
    logger.info("=" * 60)
    
    tests = [
        ("GPU Environment", test_gpu),
        ("Dataset Loading", test_dataset_loading),
        ("Model Loading", test_model_loading),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("PEFT Setup", test_peft_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ready for full experiment.")
        return True
    elif passed >= total - 1:
        logger.info("⚠️  Most tests passed. Experiment should work with minor issues.")
        return True
    else:
        logger.error("❌ Multiple test failures. Please fix issues before running experiment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
