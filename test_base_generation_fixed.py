#!/usr/bin/env python3
"""
Quick test for base model generation with the fixed generate_summary method
"""
import sys
import os
sys.path.append('/storage/nammt/billsum_sel')

from real_production_experiment import ProductionEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_base_generation():
    """Test base model generation with new implementation."""
    logger.info("Testing base model generation...")
    
    evaluator = ProductionEvaluator()
    
    # Test text
    test_text = "The bill proposes to increase funding for education by $1 billion over five years. It includes provisions for teacher training, school infrastructure improvements, and technology upgrades. The funding will be distributed to states based on student population and need."
    
    # Setup model
    if not evaluator.setup_model_and_tokenizer():
        logger.error("Failed to setup model")
        return
    
    # Generate summary
    summary = evaluator.generate_summary(test_text)
    
    logger.info(f"Generated summary: '{summary}'")
    logger.info(f"Summary length: {len(summary)}")
    logger.info(f"Summary valid: {bool(summary and len(summary) > 5)}")

if __name__ == "__main__":
    test_base_generation()
