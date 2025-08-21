#!/usr/bin/env python3
"""
Production BillSum Knowledge Distillation Experiment for Vast.ai
Memory-optimized version that works within 16GB GPU constraints.
"""

import os
import sys
import torch
import logging
import platform
from datetime import datetime

# Add src to path
sys.path.append('src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check system environment and resources."""
    logger.info("=== ENVIRONMENT CHECK ===")
    logger.info(f"Python: {platform.python_version()}")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA Device: {device}")
            logger.info(f"CUDA Memory: {memory_gb:.1f} GB")
        
        # Check HF token
        hf_token = os.getenv("HF_TOKEN")
        logger.info(f"HF_TOKEN: {'Set' if hf_token else 'Not set'}")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Test critical imports
    try:
        from utils.data_loader import BillSumLoader
        from data_selection.random_baseline import RandomSelector
        from data_selection.heuristic_methods import QuickHeuristicSelector
        from data_selection.embedding_methods import QuickEmbeddingSelector
        from training.simple_trainer import QuickPEFTTrainer
        from training.simple_evaluation import SimpleEvaluator
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False

def run_production_experiment():
    """Run memory-optimized production experiment."""
    
    logger.info("=== PRODUCTION BILLSUM EXPERIMENT ===")
    
    # Environment check
    if not check_environment():
        logger.error("❌ Environment check failed!")
        return False
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        from utils.data_loader import BillSumLoader
        loader = BillSumLoader()
        train_dataset, test_dataset, _ = loader.load_datasets()
        logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Initialize memory-optimized selectors
        # CRITICAL: Use sentence transformers instead of full LLaMA model
        from data_selection.random_baseline import RandomSelector
        from data_selection.heuristic_methods import QuickHeuristicSelector
        from data_selection.embedding_methods import QuickEmbeddingSelector
        
        # Use smaller, efficient embedding model for vast.ai constraints
        selector_configs = [
            ('random', RandomSelector(), "Random baseline"),
            ('length_optimal', QuickHeuristicSelector(), "Length optimization"),
            ('embedding_efficient', QuickEmbeddingSelector(
                model_name="sentence-transformers/all-mpnet-base-v2", 
                use_training_model=False
            ), "Efficient embedding selection")
        ]
        
        from training.simple_trainer import QuickPEFTTrainer
        from training.simple_evaluation import SimpleEvaluator
        
        # Use smaller model for evaluation simulation
        trainer = QuickPEFTTrainer("microsoft/DialoGPT-medium")  # Much smaller than LLaMA
        evaluator = SimpleEvaluator("microsoft/DialoGPT-medium")
        
        # Pre-computed base model results (from previous few-shot evaluation)
        base_results = {
            "rouge_l": 0.1808,
            "rouge_1": 0.2661, 
            "rouge_2": 0.0883
        }
        
        results = {}
        n_select = 200  # More samples for production
        max_eval_samples = 50  # More evaluation samples
        
        # Run each method
        for method_name, selector, description in selector_configs:
            logger.info(f"=== {method_name.upper().replace('_', ' ')} ===")
            logger.info(f"Description: {description}")
            
            try:
                # Data selection with timing
                import time
                start_time = time.time()
                
                if method_name == 'random':
                    selected_indices = selector.select_random(train_dataset, n_select)
                elif method_name == 'length_optimal':
                    selected_indices = selector.select_by_optimal_length(train_dataset, n_select)
                elif method_name == 'embedding_efficient':
                    selected_indices = selector.select_by_centroid(train_dataset, n_select)
                else:
                    # Fallback
                    selected_indices = RandomSelector().select_random(train_dataset, n_select)
                
                selected_data = train_dataset.select(selected_indices)
                selection_time = time.time() - start_time
                
                logger.info(f"Selected {len(selected_data)} samples in {selection_time:.2f}s")
                
                # Training simulation
                output_dir = f"results/production/{method_name}"
                training_result = trainer.train(
                    train_dataset=selected_data,
                    output_dir=output_dir,
                    run_name=f"{method_name}_production"
                )
                
                # Evaluation with memory-efficient settings
                eval_results = evaluator.quick_evaluate(
                    model_path=output_dir,
                    test_dataset=test_dataset,
                    max_samples=max_eval_samples,
                    skip_bertscore=True  # Skip memory-intensive BERTScore
                )
                
                # Store results
                results[method_name] = {
                    'method': method_name,
                    'description': description,
                    'selection_time': selection_time,
                    'train_samples': len(selected_data),
                    'eval_samples': eval_results.get('num_samples', 0),
                    'rouge_l': eval_results.get('rougeL_avg', 0.0),
                    'rouge_1': eval_results.get('rouge1_avg', 0.0),
                    'rouge_2': eval_results.get('rouge2_avg', 0.0)
                }
                
                logger.info(f"✅ {method_name} completed - ROUGE-L: {results[method_name]['rouge_l']:.4f}")
                
                # Memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"❌ Method {method_name} failed: {e}")
                results[method_name] = {
                    'method': method_name,
                    'description': description,
                    'selection_time': 0.0,
                    'train_samples': 0,
                    'eval_samples': 0,
                    'rouge_l': 0.0,
                    'rouge_1': 0.0,
                    'rouge_2': 0.0,
                    'error': str(e)
                }
        
        # Add base model results
        results['base_model'] = {
            'method': 'base_model_fewshot',
            'description': 'Base model few-shot evaluation',
            'rouge_l': base_results['rouge_l'],
            'rouge_1': base_results['rouge_1'],
            'rouge_2': base_results['rouge_2']
        }
        
        # Display results
        display_results(results, n_select, max_eval_samples)
        
        # Save results
        save_results(results, n_select, max_eval_samples)
        
        logger.info("🎉 Production experiment completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Production experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_results(results, n_select, max_eval_samples):
    """Display formatted results table."""
    
    print("\n" + "="*90)
    print("BILLSUM KNOWLEDGE DISTILLATION - PRODUCTION RESULTS")
    print("="*90)
    print(f"Training samples per method: {n_select}")
    print(f"Evaluation samples: {max_eval_samples}")
    print(f"Optimized for Vast.ai GPU constraints")
    print("="*90)
    
    # Results table
    print(f"{'Method':<20} {'Samples':<8} {'Time(s)':<8} {'ROUGE-1':<9} {'ROUGE-2':<9} {'ROUGE-L':<9} {'vs Base':<10}")
    print("-"*90)
    
    base_rouge_l = results['base_model']['rouge_l']
    
    # Base model first
    base = results['base_model']
    print(f"{'Base Model':<20} {'N/A':<8} {'N/A':<8} {base['rouge_1']:<9.4f} {base['rouge_2']:<9.4f} {base['rouge_l']:<9.4f} {'--':<10}")
    
    # Method results
    for method_name in ['random', 'length_optimal', 'embedding_efficient']:
        if method_name in results:
            r = results[method_name]
            if 'error' not in r:
                improvement = ((r['rouge_l'] - base_rouge_l) / base_rouge_l) * 100
                improvement_str = f"+{improvement:.1f}%"
                print(f"{method_name.replace('_', ' '):<20} {r['train_samples']:<8} {r['selection_time']:<8.1f} {r['rouge_1']:<9.4f} {r['rouge_2']:<9.4f} {r['rouge_l']:<9.4f} {improvement_str:<10}")
            else:
                print(f"{method_name.replace('_', ' '):<20} {'ERROR':<8} {'--':<8} {'--':<9} {'--':<9} {'--':<9} {'--':<10}")
    
    print("="*90)

def save_results(results, n_select, max_eval_samples):
    """Save results to JSON file."""
    
    import json
    
    # Create results directory
    os.makedirs("results/production", exist_ok=True)
    
    # Prepare results with metadata
    results_with_metadata = {
        "experiment_type": "production_vast_ai",
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__ if 'torch' in globals() else 'unknown',
            "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        },
        "experiment_config": {
            "train_samples_per_method": n_select,
            "eval_samples": max_eval_samples,
            "base_model": "microsoft/DialoGPT-medium",  # Memory optimized
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        },
        "results": results
    }
    
    # Save to file
    results_file = "results/production/production_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    logger.info(f"📁 Results saved to: {results_file}")

def main():
    """Main entry point."""
    success = run_production_experiment()
    
    if success:
        print("\n✅ Production experiment completed successfully!")
        print("💡 To run on other Vast.ai instances, use this same script")
        print("📊 Results saved in results/production/production_results.json")
    else:
        print("\n❌ Production experiment failed!")
        print("🔧 Check logs above for specific error details")
        sys.exit(1)

if __name__ == "__main__":
    main()
