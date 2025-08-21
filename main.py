"""
Main experiment runner for BillSum Knowledge Distillation Pipeline.
Provides quick iteration and clear comparisons between data selection strategies.
"""

import os
import argparse
import logging
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("wandb not available - experiment tracking disabled")
    WANDB_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not available - using system environment only")

class QuickExperimentRunner:
    """Run quick experiments for immediate results."""
    
    def __init__(self):
        # Import modules (with error handling)
        try:
            from utils.data_loader import BillSumLoader
            from data_selection.random_baseline import RandomSelector
            from data_selection.heuristic_methods import QuickHeuristicSelector
            from training.peft_trainer import QuickPEFTTrainer
            from training.evaluation import ComprehensiveEvaluator
            
            self.data_loader = BillSumLoader()
            self.trainer = QuickPEFTTrainer()
            self.evaluator = ComprehensiveEvaluator(self.trainer.model_name, use_comprehensive_metrics=True)
            self.random_selector = RandomSelector
            self.heuristic_selector = QuickHeuristicSelector
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Please install dependencies: pip install -r requirements.txt")
            sys.exit(1)
        
        # Initialize wandb if available
        if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
            try:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                logger.info("Wandb initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def validate_environment(self) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = ["HF_TOKEN"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Please copy .env.template to .env and fill in your tokens")
            return False
        
        return True
    
    def run_full_baseline(self) -> dict:
        """Run full dataset baseline first."""
        logger.info("=== RUNNING FULL DATASET BASELINE ===")
        
        if not self.validate_environment():
            return {}
        
        # Load full dataset
        try:
            train_data, test_data, ca_test_data = self.data_loader.load_datasets()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {}
        
        # Initialize wandb run
        run_config = {
            "method": "full_dataset",
            "train_samples": len(train_data),
            "model": self.trainer.model_name
        }
        
        if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "billsum-kd-benchmark"),
                name="full_dataset_baseline",
                config=run_config
            )
        
        # Train on full dataset
        output_dir = "results/full_baseline"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            training_result = self.trainer.train(
                train_data, 
                output_dir,
                "full_dataset_baseline"
            )
            
            # Evaluate
            results = self.evaluator.quick_evaluate(output_dir, test_data, max_samples=200)
            
            # Log results
            if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
                wandb.log(results)
                wandb.finish()
            
            # Save results
            full_results = {
                "method": "full_dataset",
                "train_samples": len(train_data),
                "results": results,
                "training_info": training_result,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(f"{output_dir}/results.json", 'w') as f:
                json.dump(full_results, f, indent=2)
            
            logger.info(f"Full baseline - ROUGE-L: {results.get('rougeL_avg', 'N/A'):.4f}")
            return full_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}
    
    def run_selection_method(self, method_name: str, selector_func, train_data, test_data, 
                           pass_training_model: bool = False) -> dict:
        """Run a single 10% selection method - with model consistency."""
        logger.info(f"=== RUNNING {method_name.upper()} SELECTION ===")
        
        n_select = int(len(train_data) * 0.1)  # 10% selection
        
        # Initialize wandb run
        run_config = {
            "method": method_name,
            "selection_percentage": 0.1,
            "train_samples": n_select,
            "total_available": len(train_data),
            "model": self.trainer.model_name
        }
        
        if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "billsum-kd-benchmark"),
                name=f"{method_name}_10percent",
                config=run_config
            )
        
        start_time = time.time()
        
        try:
            # For methods that need training model consistency
            if pass_training_model and hasattr(self.trainer, 'model') and self.trainer.model:
                selected_indices = selector_func(train_data, n_select, self.trainer.model, self.trainer.tokenizer)
            else:
                selected_indices = selector_func(train_data, n_select)
            
            selected_data = train_data.select(selected_indices)
            
            selection_time = time.time() - start_time
            logger.info(f"Selection completed in {selection_time:.2f}s")
            
            # Train
            output_dir = f"results/selection_results/{method_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            training_result = self.trainer.train(
                selected_data,
                output_dir,
                f"{method_name}_10percent"
            )
            
            # Evaluate
            results = self.evaluator.quick_evaluate(output_dir, test_data, max_samples=200)
            
            # Log results
            if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
                wandb.log({**results, "selection_time": selection_time})
                wandb.finish()
            
            # Save results
            method_results = {
                "method": method_name,
                "selection_percentage": 0.1,
                "train_samples": len(selected_data),
                "selection_time": selection_time,
                "selected_indices": selected_indices,
                "results": results,
                "training_info": training_result,
                "model_consistency": pass_training_model,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(f"{output_dir}/results.json", 'w') as f:
                json.dump(method_results, f, indent=2)
            
            logger.info(f"{method_name} - ROUGE-L: {results.get('rougeL_avg', 'N/A'):.4f}")
            return method_results
            
        except Exception as e:
            logger.error(f"Method {method_name} failed: {e}")
            return {}
    
    def run_quick_comparison(self) -> dict:
        """Run quick comparison without full baseline."""
        logger.info("=== RUNNING QUICK COMPARISON (10% METHODS ONLY) ===")
        
        if not self.validate_environment():
            return {}
        
        try:
            train_data, test_data, _ = self.data_loader.load_datasets()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {}
        
        results = {}
        
        # Random baseline
        results['random'] = self.run_selection_method(
            "random",
            lambda data, n: self.random_selector.select_random(data, n),
            train_data, test_data
        )
        
        # Length-diversity combination
        # Length-diversity combination
        results['length_diversity'] = self.run_selection_method(
            "length_diversity", 
            self.heuristic_selector.select_length_diversity_combo,
            train_data, test_data
        )
        
        # Balanced lengths method
        results['balanced_lengths'] = self.run_selection_method(
            "balanced_lengths",
            self.heuristic_selector.select_balanced_lengths,
            train_data, test_data
        )
        
        # Try embedding methods if available
        try:
            from data_selection.embedding_methods import QuickEmbeddingSelector
            embedding_selector = QuickEmbeddingSelector()
            
            results['embedding_centroid'] = self.run_selection_method(
                "embedding_centroid",
                embedding_selector.select_by_centroid,
                train_data, test_data
            )
        except Exception as e:
            logger.warning(f"Embedding methods not available: {e}")
        
        # Generate comparison report
        self.generate_quick_comparison_report(results)
        
        return results
    
    def run_all_comparisons(self) -> dict:
        """Run all selection methods for comparison with model consistency."""
        logger.info("=== LOADING DATA ===")
        
        if not self.validate_environment():
            return {}
        
        try:
            train_data, test_data, ca_test_data = self.data_loader.load_datasets()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {}
        
        # CRITICAL: Setup training model FIRST for consistent embeddings
        logger.info("=== SETTING UP TRAINING MODEL FOR CONSISTENCY ===")
        self.trainer.setup_model()
        
        all_results = {}
        
        # 1. Full baseline (if not exists)
        baseline_path = "results/full_baseline/results.json"
        if os.path.exists(baseline_path):
            logger.info("Loading existing full baseline results")
            with open(baseline_path) as f:
                all_results['full_baseline'] = json.load(f)
        else:
            all_results['full_baseline'] = self.run_full_baseline()
        
        # 2. Random baseline
        all_results['random'] = self.run_selection_method(
            "random",
            lambda data, n: self.random_selector.select_random(data, n),
            train_data, test_data
        )
        
        # 3. Length-diversity combination
        all_results['length_diversity'] = self.run_selection_method(
            "length_diversity",
            self.heuristic_selector.select_length_diversity_combo,
            train_data, test_data
        )
        
        # 4. Balanced lengths
        all_results['balanced_lengths'] = self.run_selection_method(
            "balanced_lengths",
            self.heuristic_selector.select_balanced_lengths,
            train_data, test_data
        )
        
        # 5. High information density
        all_results['high_info_density'] = self.run_selection_method(
            "high_info_density",
            self.heuristic_selector.select_high_information_density,
            train_data, test_data
        )
        
        # 6. Embedding centroid (with model consistency)
        try:
            from data_selection.embedding_methods import QuickEmbeddingSelector
            embedding_selector = QuickEmbeddingSelector(model_name=self.trainer.model_name)
            
            # First try with training model consistency
            all_results['embedding_consistent'] = self.run_selection_method(
                "embedding_consistent",
                embedding_selector.select_using_training_model_embeddings,
                train_data, test_data,
                pass_training_model=True
            )
            
            # Also run standard embedding method for comparison
            all_results['embedding_centroid'] = self.run_selection_method(
                "embedding_centroid",
                embedding_selector.select_by_centroid,
                train_data, test_data
            )
        except Exception as e:
            logger.warning(f"Embedding methods failed: {e}")
        
        # 7. Clustering-based
        try:
            if 'embedding_selector' in locals():
                all_results['clustering'] = self.run_selection_method(
                    "clustering",
                    embedding_selector.select_by_clustering,
                    train_data, test_data
                )
        except Exception as e:
            logger.warning(f"Clustering method failed: {e}")
        
        # 6. Iterative method
        try:
            from data_selection.iterative_method import IterativeSelector
            iterative_selector = IterativeSelector()
            all_results['iterative'] = self.run_selection_method(
                "iterative",
                iterative_selector.select_iterative,
                train_data, test_data
            )
        except Exception as e:
            logger.warning(f"Iterative method failed: {e}")
        
        # Generate comprehensive comparison
        self.generate_comprehensive_comparison_report(all_results)
        
        return all_results
    
    def generate_comprehensive_comparison_report(self, all_results: dict):
        """Generate comprehensive comparison report."""
        logger.info("=== GENERATING COMPREHENSIVE COMPARISON REPORT ===")
        
        comparison = {}
        baseline_rouge = all_results.get('full_baseline', {}).get('results', {}).get('rougeL_avg', 1.0)
        
        print("\n" + "="*60)
        print("BILLSUM KNOWLEDGE DISTILLATION RESULTS")
        print("="*60)
        print(f"{'Method':<20} {'ROUGE-L':<10} {'vs Full':<12} {'Samples':<10}")
        print("-"*60)
        
        for method, result in all_results.items():
            if not result:  # Skip failed methods
                continue
                
            rouge_l = result.get('results', {}).get('rougeL_avg', 0.0)
            samples = result.get('train_samples', 'N/A')
            
            if method == 'full_baseline':
                vs_full = "100.0%"
            else:
                vs_full = f"{(rouge_l/baseline_rouge)*100:.1f}%"
            
            print(f"{method:<20} {rouge_l:<10.4f} {vs_full:<12} {samples:<10}")
            
            comparison[method] = {
                'rouge_l': rouge_l,
                'vs_full_percentage': (rouge_l/baseline_rouge)*100 if method != 'full_baseline' else 100.0,
                'samples_used': samples,
                'selection_time': result.get('selection_time', 0.0)
            }
        
        print("="*60)
        
        # Save comparison
        os.makedirs("results/comparison", exist_ok=True)
        
        with open("results/comparison/summary.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def generate_quick_comparison_report(self, results: dict):
        """Generate quick comparison report."""
        logger.info("=== GENERATING COMPARISON REPORT ===")
        
        print("\n" + "="*50)
        print("QUICK COMPARISON (10% METHODS ONLY)")
        print("="*50)
        print(f"{'Method':<20} {'ROUGE-L':<10} {'Samples':<10}")
        print("-"*50)
        
        comparison = {}
        
        for method, result in results.items():
            if not result:  # Skip failed methods
                continue
                
            rouge_l = result.get('results', {}).get('rougeL_avg', 0.0)
            samples = result.get('train_samples', 'N/A')
            
            print(f"{method:<20} {rouge_l:<10.4f} {samples:<10}")
            
            comparison[method] = {
                'rouge_l': rouge_l,
                'samples_used': samples,
                'selection_time': result.get('selection_time', 0.0)
            }
        
        print("="*50)
        
        # Save comparison
        os.makedirs("results/comparison", exist_ok=True)
        
        with open("results/comparison/quick_summary.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick BillSum KD experiments")
    parser.add_argument("--mode", choices=['baseline', 'quick', 'all', 'validate'], default='quick',
                       help="Experiment mode")
    parser.add_argument("--max-train", type=int, help="Limit training data size for testing")
    
    args = parser.parse_args()
    
    runner = QuickExperimentRunner()
    
    if args.mode == 'validate':
        # Just validate environment and dataset loading
        logger.info("=== VALIDATION MODE ===")
        if runner.validate_environment():
            logger.info("✓ Environment validation passed")
            try:
                stats = runner.data_loader.get_dataset_stats()
                logger.info("✓ Dataset loading successful")
                logger.info(f"Dataset stats: {stats}")
            except Exception as e:
                logger.error(f"✗ Dataset loading failed: {e}")
        else:
            logger.error("✗ Environment validation failed")
    
    elif args.mode == 'baseline':
        runner.run_full_baseline()
    
    elif args.mode == 'quick':
        runner.run_quick_comparison()
    
    elif args.mode == 'all':
        runner.run_all_comparisons()

if __name__ == "__main__":
    main()
