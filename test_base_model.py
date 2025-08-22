#!/usr/bin/env python3
"""
Test Base Model Evaluation
Quick test to verify the base model evaluation works correctly
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_base_model_evaluation():
    """Test base model evaluation specifically."""
    logger.info("🧪 Testing Base Model Evaluation")
    logger.info("=" * 50)
    
    try:
        # Import classes
        from real_production_experiment import ProductionEvaluator, load_dataset
        
        # Load small dataset
        logger.info("📊 Loading test dataset...")
        train_dataset, test_dataset = load_dataset()
        
        # Use only 3 samples for quick test
        mini_test = test_dataset.select(range(3))
        logger.info(f"✅ Test dataset: {len(mini_test)} samples")
        
        # Test base model evaluation
        logger.info("🤖 Testing base model evaluation...")
        evaluator = ProductionEvaluator()
        
        # Test with just 1 sample for detailed debugging
        result = evaluator.evaluate_base_model(mini_test, max_samples=1)
        
        if result and "bertscore_f1" in result:
            logger.info("✅ Base model evaluation successful!")
            logger.info(f"   📊 BERTScore F1: {result['bertscore_f1']:.4f}")
            logger.info(f"   📊 ROUGE-Lsum: {result.get('rougeLsum', 0.0):.4f}")
            logger.info(f"   📊 ROUGE-1: {result.get('rouge1', 0.0):.4f}")
            logger.info(f"   📊 Samples evaluated: {result.get('num_samples', 0)}")
            
            # Show sample prediction
            if "predictions" in result and result["predictions"]:
                logger.info(f"   📝 Sample prediction: {result['predictions'][0][:100]}...")
            
            logger.info("=" * 50)
            logger.info("🎉 Base model evaluation test PASSED!")
            return True
        else:
            logger.error("❌ Base model evaluation failed")
            logger.error(f"Result: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure results directory exists
    Path("results/production").mkdir(parents=True, exist_ok=True)
    
    success = test_base_model_evaluation()
    sys.exit(0 if success else 1)
