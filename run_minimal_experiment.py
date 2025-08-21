#!/usr/bin/env python3
"""
Memory-optimized quick experiment runner.
Runs a minimal version of the experiment to demonstrate the pipeline functionality.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_minimal_experiment():
    """Run minimal experiment with very small samples and simplified evaluation."""
    
    # Set tokens (get from environment variables)
    # os.environ['HF_TOKEN'] = 'your_huggingface_token_here'  # Set this in your environment
    # os.environ['WANDB_API_KEY'] = 'your_wandb_key_here'     # Optional for tracking
    
    try:
        # Import components
        from src.utils.data_loader import BillSumLoader
        from src.data_selection.random_baseline import RandomSelector
        from src.data_selection.heuristic_methods import QuickHeuristicSelector
        from src.data_selection.embedding_methods import QuickEmbeddingSelector
        from src.training.simple_trainer import QuickPEFTTrainer
        from src.training.simple_evaluation import SimpleEvaluator
        
        logger.info("=== MINIMAL BILLSUM EXPERIMENT ===")
        
        # 1. Load dataset
        logger.info("Loading dataset...")
        loader = BillSumLoader()
        train_dataset, test_dataset, _ = loader.load_datasets()
        logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Use very small sample sizes
        n_select = 100  # Very small selection
        test_samples = 20  # Very small test set
        
        # Initialize components
        random_selector = RandomSelector()
        heuristic_selector = QuickHeuristicSelector()
        embedding_selector = QuickEmbeddingSelector(model_name="meta-llama/Llama-2-7b-hf", use_training_model=False)  # Use sentence transformers
        trainer = QuickPEFTTrainer()
        evaluator = SimpleEvaluator(base_model_name="meta-llama/Llama-2-7b-hf")
        
        # Get a small test set
        test_subset = test_dataset.select(range(test_samples))
        
        results = {}
        
        # Method 1: Random Selection
        logger.info("=== RANDOM SELECTION ===")
        start_time = time.time()
        random_indices = random_selector.select_random(train_dataset, n_select)
        random_data = train_dataset.select(random_indices)
        selection_time = time.time() - start_time
        
        # Train
        trainer.train(
            train_dataset=random_data,
            output_dir="results/minimal/random",
            run_name="random_minimal"
        )
        
        # Evaluate with simplified metrics (no BERTScore to save memory)
        eval_results = evaluator.quick_evaluate(
            model_path="results/minimal/random",
            test_dataset=test_subset,
            max_samples=test_samples,
            skip_bertscore=True  # Skip memory-intensive BERTScore
        )
        
        results['random'] = {
            'method': 'random',
            'selection_time': selection_time,
            'train_samples': len(random_data),
            'rouge_l': eval_results.get('rougeL_avg', 0.0),
            'rouge_1': eval_results.get('rouge1_avg', 0.0),
            'rouge_2': eval_results.get('rouge2_avg', 0.0)
        }
        logger.info(f"Random - ROUGE-L: {results['random']['rouge_l']:.4f}")
        
        # Method 2: Length-based Selection
        logger.info("=== LENGTH-BASED SELECTION ===")
        start_time = time.time()
        length_indices = heuristic_selector.select_by_optimal_length(train_dataset, n_select)
        length_data = train_dataset.select(length_indices)
        selection_time = time.time() - start_time
        
        # Train
        trainer.train(
            train_dataset=length_data,
            output_dir="results/minimal/length",
            run_name="length_minimal"
        )
        
        # Evaluate
        eval_results = evaluator.quick_evaluate(
            model_path="results/minimal/length",
            test_dataset=test_subset,
            max_samples=test_samples,
            skip_bertscore=True
        )
        
        results['length'] = {
            'method': 'length_based',
            'selection_time': selection_time,
            'train_samples': len(length_data),
            'rouge_l': eval_results.get('rougeL_avg', 0.0),
            'rouge_1': eval_results.get('rouge1_avg', 0.0),
            'rouge_2': eval_results.get('rouge2_avg', 0.0)
        }
        logger.info(f"Length - ROUGE-L: {results['length']['rouge_l']:.4f}")
        
        # Method 3: Embedding-based Selection (using sentence transformers)
        logger.info("=== EMBEDDING-BASED SELECTION ===")
        start_time = time.time()
        embedding_indices = embedding_selector.select_by_centroid(train_dataset, n_select)
        embedding_data = train_dataset.select(embedding_indices)
        selection_time = time.time() - start_time
        
        # Train  
        trainer.train(
            train_dataset=embedding_data,
            output_dir="results/minimal/embedding",
            run_name="embedding_minimal"
        )
        
        # Evaluate
        eval_results = evaluator.quick_evaluate(
            model_path="results/minimal/embedding",
            test_dataset=test_subset,
            max_samples=test_samples,
            skip_bertscore=True
        )
        
        results['embedding'] = {
            'method': 'embedding_centroid',
            'selection_time': selection_time,
            'train_samples': len(embedding_data),
            'rouge_l': eval_results.get('rougeL_avg', 0.0),
            'rouge_1': eval_results.get('rouge1_avg', 0.0),
            'rouge_2': eval_results.get('rouge2_avg', 0.0)
        }
        logger.info(f"Embedding - ROUGE-L: {results['embedding']['rouge_l']:.4f}")
        
        # Load base model results if available
        base_results_file = "results/base_model_fewshot_1shot/results_base_model.json"
        if os.path.exists(base_results_file):
            with open(base_results_file, 'r') as f:
                base_results = json.load(f)
            results['base_model'] = {
                'method': 'base_model_fewshot', 
                'rouge_l': base_results.get('rougeL', 0.0),
                'rouge_1': base_results.get('rouge1', 0.0),
                'rouge_2': base_results.get('rouge2', 0.0)
            }
        
        # Save complete results
        output_dir = Path("results/minimal")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        final_results = {
            'experiment_type': 'minimal_comparison',
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_train': len(train_dataset),
                'total_test': len(test_dataset),
                'selected_per_method': n_select,
                'test_samples': test_samples
            },
            'results': results
        }
        
        with open(output_dir / "experiment_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print results table
        print("\n" + "="*80)
        print("BILLSUM KNOWLEDGE DISTILLATION EXPERIMENT RESULTS")
        print("="*80)
        print(f"{'Method':<20} {'Train Samples':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Time (s)':<10}")
        print("-"*80)
        
        if 'base_model' in results:
            print(f"{'Base Model':<20} {'N/A':<15} {results['base_model']['rouge_1']:<10.4f} {results['base_model']['rouge_2']:<10.4f} {results['base_model']['rouge_l']:<10.4f} {'N/A':<10}")
        
        for method_name, method_results in results.items():
            if method_name != 'base_model':
                print(f"{method_results['method']:<20} {method_results['train_samples']:<15} {method_results['rouge_1']:<10.4f} {method_results['rouge_2']:<10.4f} {method_results['rouge_l']:<10.4f} {method_results.get('selection_time', 0):<10.2f}")
        
        print("="*80)
        print("Experiment completed successfully!")
        print(f"Results saved to: {output_dir}/experiment_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_minimal_experiment()
    if results:
        print("\n🎉 Minimal experiment completed successfully!")
    else:
        print("\n❌ Experiment failed!")
        sys.exit(1)
