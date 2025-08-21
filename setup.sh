#!/bin/bash

# BillSum Knowledge Distillation Pipeline Setup Script
echo "🚀 Setting up BillSum Knowledge Distillation Pipeline..."

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your tokens before running experiments"
fi

# Validate setup
echo "✅ Running validation..."
python main.py --mode validate

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Edit .env file with your HuggingFace and WandB tokens"
echo "2. Run: python main.py --mode quick    # For 2-3 hour test"
echo "3. Run: python main.py --mode full     # For complete experiment"
echo ""
