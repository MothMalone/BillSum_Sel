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
            from src.utils.data_loader import BillSumLoader
            from src.data_selection.random_baseline import RandomSelector
            from src.data_selection.heuristic_methods import QuickHeuristicSelector
            from src.training.simple_trainer import QuickPEFTTrainer
            from src.training.simple_evaluation import SimpleEvaluator
            
            self.data_loader = BillSumLoader()
            self.trainer = QuickPEFTTrainer()
            self.evaluator = SimpleEvaluator(self.trainer.model_name, use_comprehensive_metrics=True)
            self.random_selector = RandomSelector
            self.heuristic_selector = QuickHeuristicSelector
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Please install dependencies: pip install -r requirements.txt")
            sys.exit(1)
        
        # Initialize wandb if available
        global WANDB_AVAILABLE
        if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
            try:
                # Check if already logged in, if not login
                if not wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=False):
                    logger.warning("WandB login failed - experiment tracking disabled")
                    WANDB_AVAILABLE = False
                else:
                    logger.info("WandB initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                logger.warning("Experiment tracking disabled")
                # Disable wandb for this session
                WANDB_AVAILABLE = False
    
    def validate_environment(self) -> bool:
        """Validate environment and prompt for tokens interactively."""
        logger.info("🔍 Validating environment setup...")
        
        # Check if .env exists, create if not
        if not os.path.exists('.env'):
            logger.info("📝 Creating .env file from template...")
            import shutil
            shutil.copy('.env.template', '.env')
        
        # Interactive token input (safer than storing in files)
        hf_token = os.getenv('HF_TOKEN')
        
        # Check if token is placeholder or empty
        if not hf_token or hf_token == 'your_huggingface_token_here' or hf_token.startswith('hf_dummy'):
            print("\n🔑 HuggingFace Token Required")
            print("This is needed to download models and datasets.")
            print("Get your token from: https://huggingface.co/settings/tokens")
            hf_token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
            
            if hf_token:
                # Set for current session only (don't save to file)
                os.environ['HF_TOKEN'] = hf_token
                logger.info("✅ HuggingFace token set for this session")
            else:
                logger.warning("⚠️  No HuggingFace token provided - some features may not work")
        else:
            logger.info("✅ HuggingFace token found")
        
        # Optional WandB token
        wandb_token = os.getenv('WANDB_API_KEY')
        if not wandb_token or wandb_token == 'your_wandb_key_here':
            print("\n📊 WandB API Key (Optional)")
            print("This enables experiment tracking and visualization.")
            print("Get your key from: https://wandb.ai/authorize")
            wandb_token = input("Enter your WandB API key (or press Enter to skip): ").strip()
            
            if wandb_token:
                os.environ['WANDB_API_KEY'] = wandb_token
                logger.info("✅ WandB API key set for this session")
            else:
                logger.info("ℹ️  WandB tracking disabled - experiment tracking will be local only")
        else:
            logger.info("✅ WandB API key found")
        
        # Test basic imports
        try:
            logger.info("🧪 Testing core dependencies...")
            import torch
            import transformers
            import datasets
            logger.info(f"✅ PyTorch: {torch.__version__}")
            logger.info(f"✅ Transformers: {transformers.__version__}")
            logger.info(f"✅ Datasets: {datasets.__version__}")
        except ImportError as e:
            logger.error(f"❌ Dependency error: {e}")
            return False
        
        # Test our custom modules
        try:
            logger.info("🧪 Testing pipeline components...")
            from src.utils.metrics import MetricsCalculator
            calc = MetricsCalculator(device='cpu')
            logger.info("✅ Metrics calculator working")
            
            # Quick test
            test_pred = ["The bill establishes new standards."]
            test_ref = ["This legislation creates new regulations."]
            metrics = calc.calculate_all_metrics(test_pred, test_ref)
            logger.info(f"✅ Sample ROUGE-L: {metrics.get('rougeL_avg', 0):.3f}")
            logger.info(f"✅ Sample BERTScore: {metrics.get('bert_score_f1', 0):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline component error: {e}")
            return False
        
        logger.info("✅ Environment validation completed successfully!")
        print("\n🎉 Setup Complete! You can now run:")
        print("   python main.py --mode quick    # 2-3 hour experiment")
        print("   python main.py --mode full     # Complete experiment")
        
        return True
    
    def evaluate_base_model(self, test_data, num_samples=50) -> dict:
        """Evaluate the base model with zero-shot prompting."""
        logger.info("=== EVALUATING BASE MODEL (FEW-SHOT) ===")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            from src.utils.metrics import MetricsCalculator
            
            # Load base model
            model_name = self.trainer.model_name
            logger.info(f"Loading base model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model.eval()
            
            # Take a sample of test data
            sample_data = test_data.select(range(min(num_samples, len(test_data))))
            
            predictions = []
            references = []
            
            logger.info(f"Generating summaries for {len(sample_data)} samples...")
            
            for i, example in enumerate(sample_data):
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(sample_data)}")
                
                # Create zero-shot prompt
                prompt = f"""Summarize the following bill text in one concise sentence:

Bill: {example['text'][:1000]}...

Summary:"""
                
                # Generate
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Extract generated text
                generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                prediction = generated.split('\n')[0].strip()
                
                predictions.append(prediction)
                references.append(example['summary'])
            
            # Calculate metrics
            metrics_calc = MetricsCalculator(device='cpu')
            metrics = metrics_calc.calculate_all_metrics(predictions, references)
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()
            
            logger.info(f"Base Model Performance:")
            logger.info(f"  ROUGE-L: {metrics.get('rougeL_avg', 0):.4f}")
            logger.info(f"  BERTScore-F1: {metrics.get('bert_score_f1', 0):.4f}")
            
            return {
                'method': 'base_model',
                'rouge_l': metrics.get('rougeL_avg', 0),
                'bert_score_f1': metrics.get('bert_score_f1', 0),
                'samples': len(sample_data),
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"Base model evaluation failed: {e}")
            return {
                'method': 'base_model',
                'rouge_l': 0.0,
                'bert_score_f1': 0.0,
                'samples': 0,
                'error': str(e)
            }
    
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
        
        wandb_run = None
        if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
            try:
                wandb_run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "billsum-kd-benchmark"),
                    name="full_dataset_baseline",
                    config=run_config
                )
                logger.info(f"WandB run initialized: {wandb_run.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB run: {e}")
                logger.warning("Continuing without WandB logging...")
        
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
            if wandb_run is not None:
                try:
                    wandb.log(results)
                    wandb.finish()
                    logger.info("Results logged to WandB successfully")
                except Exception as e:
                    logger.warning(f"Failed to log results to WandB: {e}")
                    try:
                        wandb.finish()
                    except:
                        pass
            
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
        
        wandb_run = None
        if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
            try:
                wandb_run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "billsum-kd-benchmark"),
                    name=f"{method_name}_10percent",
                    config=run_config
                )
                logger.info(f"WandB run initialized: {wandb_run.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB run: {e}")
                logger.warning("Continuing without WandB logging...")
        
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
            if wandb_run is not None:
                try:
                    wandb.log({**results, "selection_time": selection_time})
                    wandb.finish()
                    logger.info("Results logged to WandB successfully")
                except Exception as e:
                    logger.warning(f"Failed to log results to WandB: {e}")
                    try:
                        wandb.finish()
                    except:
                        pass
            
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
        
        # Base model evaluation (use existing results if available)
        logger.info("Checking for existing base model results...")
        base_results_file = "results/base_model_fewshot_1shot/results_base_model.json"
        if os.path.exists(base_results_file):
            logger.info("Using existing base model evaluation results...")
            with open(base_results_file, 'r') as f:
                base_results = json.load(f)
            results['base_model'] = {
                'method': 'base_model_fewshot',
                'rouge_l': base_results.get('rougeL', 0.0),
                'bert_score_f1': base_results.get('bertscore_f1', 0.0),
                'samples': 100,
                'model_name': self.trainer.model_name
            }
            logger.info(f"Base model - ROUGE-L: {results['base_model']['rouge_l']:.4f}")
        else:
            logger.info("No existing base model results found, skipping for memory efficiency...")
            results['base_model'] = {
                'method': 'base_model_skipped',
                'rouge_l': 0.0,
                'bert_score_f1': 0.0,
                'samples': 0,
                'note': 'Skipped due to memory constraints'
            }
        
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
            from src.data_selection.embedding_methods import QuickEmbeddingSelector
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
            from src.data_selection.embedding_methods import QuickEmbeddingSelector
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
            from src.data_selection.iterative_method import IterativeSelector
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
