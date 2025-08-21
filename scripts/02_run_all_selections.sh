#!/bin/bash

# Run all data selection methods (10% each)
# Comprehensive comparison of all selection strategies

set -e

echo "=== BillSum All Selection Methods Runner ==="
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
echo "🚀 Running all selection methods..."
echo "📊 Each method uses 10% of training data"
echo "⏱️  Estimated time: 3-4 hours for all methods"
echo ""
echo "Methods to run:"
echo "  1. Random baseline"
echo "  2. Stratified random"
echo "  3. Optimal length"
echo "  4. Diversity selection"
echo "  5. Length-diversity combo"
echo "  6. Summary ratio"
echo "  7. Embedding centroid"
echo "  8. Clustering-based"
echo "  9. Diversity sampling"
echo "  10. Iterative pruning"
echo ""

# Confirm with user
read -p "Continue with all selection methods? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

start_time=$(date +%s)

# Run comprehensive selection comparison
python main.py --mode all

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "✅ All selection methods completed!"
echo "⏱️  Total time: $(date -u -d @${duration} +%H:%M:%S)"
echo ""
echo "📊 Results saved to:"
echo "  - results/selection_results/*/results.json"
echo "  - results/comparison/"
echo ""

# Generate final comparison
echo "📈 Generating comprehensive comparison report..."
python scripts/03_generate_comparison.py

echo ""
echo "🎯 Check results/comparison/ for:"
echo "  - summary.json: Detailed comparison data"
echo "  - *.png: Visualization plots"
echo "  - summary_table.csv: Excel-friendly results"
echo ""
echo "✅ Complete pipeline finished!"
