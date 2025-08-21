#!/bin/bash

# Quick Setup & Validation Script for BillSum Knowledge Distillation
# This script sets up the environment and validates everything works

set -e  # Exit on any error

echo "🚀 BillSum Knowledge Distillation - Quick Setup & Validation"
echo "============================================================"

# Check Python
echo "🐍 Checking Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env and add your tokens:"
    echo "   - HF_TOKEN=your_huggingface_token"
    echo "   - WANDB_API_KEY=your_wandb_key (optional)"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if HF_TOKEN is set
if ! grep -q "HF_TOKEN=hf_" .env; then
    echo "⚠️  HF_TOKEN not properly set in .env file"
    echo "Please add your Hugging Face token to .env:"
    echo "   HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "✅ Environment file configured"

# Install core dependencies first
echo "📦 Installing core dependencies..."
python3 -m pip install --quiet --upgrade pip

# Install in stages to avoid conflicts
echo "   Installing PyTorch..."
python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo "   Installing transformers and datasets..."
python3 -m pip install --quiet transformers datasets accelerate

echo "   Installing PEFT and sentence transformers..."
python3 -m pip install --quiet peft sentence-transformers

echo "   Installing evaluation and utilities..."
python3 -m pip install --quiet rouge-score scikit-learn python-dotenv nltk

echo "   Installing optional dependencies..."
python3 -m pip install --quiet wandb matplotlib seaborn tqdm psutil --quiet || echo "Some optional packages failed to install"

echo "✅ Dependencies installed"

# Test imports
echo "🧪 Testing critical imports..."
python3 -c "
import sys
failed = []

try:
    import torch
    print('✅ PyTorch')
except ImportError:
    failed.append('torch')
    print('❌ PyTorch')

try:
    import transformers
    print('✅ Transformers')
except ImportError:
    failed.append('transformers')
    print('❌ Transformers')

try:
    import datasets
    print('✅ Datasets')
except ImportError:
    failed.append('datasets')
    print('❌ Datasets')

try:
    import peft
    print('✅ PEFT')
except ImportError:
    failed.append('peft')
    print('❌ PEFT')

if failed:
    print(f'❌ Failed imports: {failed}')
    sys.exit(1)
else:
    print('✅ All critical imports successful')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed!"
    exit 1
fi

# Test dataset access
echo "🔍 Testing dataset access..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from datasets import load_dataset
    print('Testing dataset access with your token...')
    
    # Test with streaming to avoid downloading everything
    dataset = load_dataset('MothMalone/SLMS-KD-Benchmarks', name='billsum', streaming=True, use_auth_token=os.getenv('HF_TOKEN'))
    
    # Try to get one sample
    train_sample = next(iter(dataset['train']))
    print(f'✅ Dataset accessible')
    print(f'   Sample keys: {list(train_sample.keys())}')
    
    # Check for required columns
    if 'text' in train_sample and ('summary' in train_sample or 'sum' in train_sample):
        print('✅ Required columns found')
    else:
        print('⚠️  Column names may need adjustment')
        
except Exception as e:
    print(f'❌ Dataset access failed: {e}')
    print('This could be due to:')
    print('1. Invalid HF_TOKEN')
    print('2. Network issues') 
    print('3. Dataset access restrictions')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dataset test failed!"
    exit 1
fi

# Test basic model loading
echo "🤖 Testing model loading (small model)..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    model_name = 'gpt2'  # Small model for testing
    print(f'Testing model: {model_name}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32  # CPU compatible
    )
    
    print(f'✅ Model loaded successfully')
    print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Model test failed!"
    exit 1
fi

echo ""
echo "🎉 VALIDATION COMPLETE!"
echo "======================="
echo "✅ Environment configured"
echo "✅ Dependencies installed"  
echo "✅ Dataset accessible"
echo "✅ Model loading works"
echo ""
echo "🚀 Ready to run experiments:"
echo "   Quick test (5 min):     python3 main.py --mode validate"
echo "   Quick comparison (30m): python3 main.py --mode quick"
echo "   Full analysis (4-6h):   python3 main.py --mode all"
echo ""
echo "📊 Or use the scripts:"
echo "   ./scripts/quick_experiment.sh"
echo ""

# Create a simple test run
echo "🧪 Running validation mode as final test..."
python3 main.py --mode validate

echo "✅ Setup validation complete! You're ready to go!"
