#!/bin/bash

# Full dataset baseline runner
# Establishes performance ceiling with complete training data

set -e

echo "=== BillSum Full Dataset Baseline Runner ==="
echo "Starting $(date)"

# Check environment
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please copy .env.template to .env and fill tokens."
    exit 1
fi

echo "🔍 Validating environment..."
python main.py --mode validate

if [ $? -ne 0 ]; then
    echo "❌ Environment validation failed!"
    exit 1
fi

echo "✅ Environment validated"
echo ""
echo "🚀 Starting full dataset baseline training..."
echo "⚠️  This will take 4-6 hours with complete BillSum dataset"
echo "📊 This establishes the performance ceiling for all comparisons"
echo ""

# Confirm with user
read -p "Continue with full baseline training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

start_time=$(date +%s)

# Run full baseline
python main.py --mode baseline

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "✅ Full baseline training completed!"
echo "⏱️  Total time: $(date -u -d @${duration} +%H:%M:%S)"
echo ""
echo "📊 Results saved to:"
echo "  - results/full_baseline/results.json"
echo "  - Model: results/full_baseline/"
echo ""

# Display results if available
if [ -f "results/full_baseline/results.json" ]; then
    echo "🎯 Baseline Results:"
    python -c "
import json
with open('results/full_baseline/results.json') as f:
    data = json.load(f)
    results = data['results']
    print(f\"ROUGE-1: {results.get('rouge1_avg', 0):.4f}\")
    print(f\"ROUGE-2: {results.get('rouge2_avg', 0):.4f}\") 
    print(f\"ROUGE-L: {results.get('rougeL_avg', 0):.4f}\")
    print(f\"Training samples: {data.get('train_samples', 'N/A')}\")
"
fi

echo ""
echo "🔄 Next steps:"
echo "  - Run selection comparisons: ./scripts/run_all_selections.sh"
echo "  - Generate final comparison: python scripts/generate_comparison.py"
