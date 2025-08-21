#!/bin/bash

# Quick 30-minute experiment runner
# Tests 3 data selection methods with 10% data each

set -e  # Exit on any error

echo "=== BillSum Quick Experiment Runner ==="
echo "Starting $(date)"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.template to .env and fill in your tokens:"
    echo "  cp .env.template .env"
    echo "  # Edit .env with your HF_TOKEN and WANDB_API_KEY"
    exit 1
fi

# Validate environment first
echo "🔍 Validating environment..."
python main.py --mode validate

if [ $? -ne 0 ]; then
    echo "❌ Environment validation failed!"
    exit 1
fi

echo "✅ Environment validation passed"

# Run quick comparison
echo "🚀 Starting quick comparison (3 methods, ~30 minutes)..."
echo "This will:"
echo "  1. Load dataset (MothMalone/SLMS-KD-Benchmarks)"
echo "  2. Run random baseline (10% data)"
echo "  3. Run length-diversity selection (10% data)"
echo "  4. Run embedding centroid selection (10% data)"
echo "  5. Generate comparison report"

start_time=$(date +%s)

python main.py --mode quick

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "✅ Quick experiment completed!"
echo "⏱️  Total time: $(date -u -d @${duration} +%H:%M:%S)"
echo ""
echo "📊 Results saved to:"
echo "  - results/selection_results/*/results.json"
echo "  - results/comparison/quick_summary.json"
echo ""
echo "📈 Check the comparison report above for immediate insights!"

# Display quick summary if results exist
if [ -f "results/comparison/quick_summary.json" ]; then
    echo ""
    echo "🎯 Quick Summary:"
    python -c "
import json
with open('results/comparison/quick_summary.json') as f:
    data = json.load(f)
    
print('Method               ROUGE-L    Time(s)')
print('-' * 40)
for method, stats in data.items():
    rouge = stats.get('rouge_l', 0)
    time_s = stats.get('selection_time', 0)
    print(f'{method:<20} {rouge:<10.4f} {time_s:<8.1f}')
"
fi

echo ""
echo "🔄 Next steps:"
echo "  - Run full baseline: ./scripts/run_full_baseline.sh"
echo "  - Analyze results: jupyter notebook analysis/quick_analysis.ipynb"
echo "  - Scale up: python main.py --mode all"
