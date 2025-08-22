#!/usr/bin/env python3
"""
Test LoRA Setup
Quick test to verify LoRA is properly configured and trainable parameters exist
"""

import logging
import sys
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lora_setup():
    """Test LoRA setup specifically."""
    logger.info("🧪 Testing LoRA Setup")
    logger.info("=" * 50)
    
    try:
        # Import classes
        from real_production_experiment import MemoryOptimizedPEFTTrainer
        
        # Test trainer setup
        logger.info("🏋️ Creating trainer...")
        trainer = MemoryOptimizedPEFTTrainer()
        
        # Test model setup
        logger.info("🔧 Setting up model with LoRA...")
        success = trainer.setup_model_and_tokenizer()
        
        if success and trainer.model is not None:
            logger.info("✅ Model setup successful!")
            
            # Check trainable parameters
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in trainer.model.parameters())
            
            logger.info(f"📊 Total parameters: {total_params:,}")
            logger.info(f"📊 Trainable parameters: {trainable_params:,}")
            logger.info(f"📊 Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            
            # Check if any parameters require gradients
            requires_grad_params = [name for name, param in trainer.model.named_parameters() if param.requires_grad]
            logger.info(f"📊 Parameters requiring gradients: {len(requires_grad_params)}")
            
            if len(requires_grad_params) > 0:
                logger.info("   Sample trainable parameters:")
                for i, name in enumerate(requires_grad_params[:5]):  # Show first 5
                    logger.info(f"     - {name}")
                if len(requires_grad_params) > 5:
                    logger.info(f"     ... and {len(requires_grad_params) - 5} more")
            
            # Check model mode
            logger.info(f"📊 Model training mode: {trainer.model.training}")
            
            # Clear memory immediately after test
            del trainer.model
            del trainer.tokenizer
            torch.cuda.empty_cache()
            
            if trainable_params > 0:
                logger.info("=" * 50)
                logger.info("🎉 LoRA setup test PASSED!")
                return True
            else:
                logger.error("❌ No trainable parameters found!")
                return False
        else:
            logger.error("❌ Model setup failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lora_setup()
    sys.exit(0 if success else 1)
