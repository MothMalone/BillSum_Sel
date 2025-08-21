#!/usr/bin/env python3
"""
BillSum Results Summary - Focus on Main Metrics
Displays BERTScore F1 and ROUGE-Lsum results in a clean format
"""

import json
import sys
from pathlib import Path

def load_results():
    """Load results from the production experiment."""
    results_file = Path("results/production/real_experiment_results.json")
    
    if not results_file.exists():
        print("❌ No results file found. Run the experiment first:")
        print("   python real_production_experiment.py")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def display_main_metrics_summary(data):
    """Display focused summary of main metrics."""
    print("\n" + "="*80)
    print("🎯 BILLSUM KNOWLEDGE DISTILLATION - MAIN METRICS SUMMARY")
    print("="*80)
    
    if "results" not in data:
        print("❌ Invalid results format")
        return
    
    results = data["results"]
    
    # Header
    print(f"\n{'Method':<20} {'BERTScore F1':<15} {'ROUGE-Lsum':<15} {'Status'}")
    print("-" * 65)
    
    # Track best scores
    best_bertscore = 0.0
    best_rougeLsum = 0.0
    best_bertscore_method = ""
    best_rougeLsum_method = ""
    
    # Results for each method
    for method_name, result in results.items():
        method_display = method_name.replace("_", "-").title()
        
        if "error" in result:
            print(f"{method_display:<20} {'---':<15} {'---':<15} ❌ Failed")
            continue
            
        if "evaluation" not in result:
            print(f"{method_display:<20} {'---':<15} {'---':<15} ⚠️  Partial")
            continue
            
        eval_data = result["evaluation"]
        bertscore = eval_data.get("bertscore_f1", 0.0)
        rougeLsum = eval_data.get("rougeLsum", eval_data.get("rougeL", 0.0))
        
        # Track best scores
        if bertscore > best_bertscore:
            best_bertscore = bertscore
            best_bertscore_method = method_display
            
        if rougeLsum > best_rougeLsum:
            best_rougeLsum = rougeLsum
            best_rougeLsum_method = method_display
        
        print(f"{method_display:<20} {bertscore:<15.4f} {rougeLsum:<15.4f} ✅ Success")
    
    # Best performers
    print("\n" + "="*80)
    print("🏆 BEST PERFORMERS")
    print("="*80)
    print(f"🥇 Best BERTScore F1:  {best_bertscore_method} ({best_bertscore:.4f})")
    print(f"🥇 Best ROUGE-Lsum:   {best_rougeLsum_method} ({best_rougeLsum:.4f})")
    
    # Experiment info
    if "config" in data:
        config = data["config"]
        print(f"\n📊 Experiment Configuration:")
        print(f"   Training Samples: {config.get('train_samples', 'N/A')}")
        print(f"   Evaluation Samples: {config.get('eval_samples', 'N/A')}")
        print(f"   Real Training: {config.get('real_training', 'N/A')}")
        print(f"   GPU Optimized: {config.get('gpu_optimized', 'N/A')}")
    
    if "timestamp" in data:
        print(f"   Timestamp: {data['timestamp']}")
    
    print("\n" + "="*80)

def display_detailed_comparison(data):
    """Display detailed comparison including all metrics."""
    print("\n📈 DETAILED METRICS COMPARISON")
    print("="*80)
    
    results = data["results"]
    
    for method_name, result in results.items():
        if "evaluation" not in result or "error" in result:
            continue
            
        method_display = method_name.replace("_", "-").title()
        eval_data = result["evaluation"]
        
        print(f"\n{method_display}:")
        print(f"  🎯 BERTScore F1:     {eval_data.get('bertscore_f1', 0.0):.4f}")
        print(f"  🎯 ROUGE-Lsum:      {eval_data.get('rougeLsum', eval_data.get('rougeL', 0.0)):.4f}")
        print(f"  📊 ROUGE-1:         {eval_data.get('rouge1', 0.0):.4f}")
        print(f"  📊 ROUGE-2:         {eval_data.get('rouge2', 0.0):.4f}")
        print(f"  📊 ROUGE-L:         {eval_data.get('rougeL', 0.0):.4f}")
        
        if 'bertscore_precision' in eval_data:
            print(f"  📊 BERTScore P:      {eval_data['bertscore_precision']:.4f}")
            print(f"  📊 BERTScore R:      {eval_data['bertscore_recall']:.4f}")
        
        print(f"  ⏱️  Training Time:    {result.get('training_time', 0.0):.1f}s")
        print(f"  📦 Dataset Size:     {result.get('dataset_size', 'N/A')}")

def main():
    """Main function."""
    data = load_results()
    if data is None:
        sys.exit(1)
    
    # Main metrics summary
    display_main_metrics_summary(data)
    
    # Detailed comparison (optional)
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        display_detailed_comparison(data)
    else:
        print("\n💡 For detailed metrics, run: python show_results.py --detailed")

if __name__ == "__main__":
    main()
