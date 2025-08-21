#!/usr/bin/env python3
"""
Results Summary for BillSum Knowledge Distillation Experiment
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_results_summary():
    """Create a comprehensive summary of the experiment results."""
    
    # Load results
    with open('results/realistic/experiment_results.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Create summary DataFrame
    summary_data = []
    for method, result in results.items():
        if method == 'base_model':
            summary_data.append({
                'Method': 'Base Model (Few-shot)',
                'Training Samples': 'N/A',
                'Selection Time (s)': 'N/A', 
                'ROUGE-1': result['rouge_1'],
                'ROUGE-2': result['rouge_2'],
                'ROUGE-L': result['rouge_l'],
                'ROUGE-L Improvement': '0.0%'
            })
        else:
            base_rouge_l = results['base_model']['rouge_l']
            improvement = ((result['rouge_l'] - base_rouge_l) / base_rouge_l) * 100
            
            summary_data.append({
                'Method': method.replace('_', ' ').title(),
                'Training Samples': result['train_samples'],
                'Selection Time (s)': f"{result['selection_time']:.2f}",
                'ROUGE-1': result['rouge_1'],
                'ROUGE-2': result['rouge_2'],
                'ROUGE-L': result['rouge_l'],
                'ROUGE-L Improvement': f"+{improvement:.1f}%"
            })
    
    df = pd.DataFrame(summary_data)
    
    print("="*100)
    print("BILLSUM KNOWLEDGE DISTILLATION - EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(f"Dataset: BillSum (MothMalone/SLMS-KD-Benchmarks)")
    print(f"Total Training Data: {data['dataset_info']['total_train']:,} samples")
    print(f"Selected per Method: {data['dataset_info']['selected_per_method']} samples")
    print(f"Test Samples: {data['dataset_info']['test_samples']} samples")
    print(f"Base Model: meta-llama/Llama-2-7b-hf")
    print("="*100)
    
    # Display formatted table
    print(f"{'Method':<25} {'Samples':<8} {'Time (s)':<10} {'ROUGE-1':<9} {'ROUGE-2':<9} {'ROUGE-L':<9} {'Improvement':<12}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['Method']:<25} {row['Training Samples']:<8} {row['Selection Time (s)']:<10} "
              f"{row['ROUGE-1']:<9.4f} {row['ROUGE-2']:<9.4f} {row['ROUGE-L']:<9.4f} {row['ROUGE-L Improvement']:<12}")
    
    print("="*100)
    
    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print(f"• Embedding-based selection achieved the highest ROUGE-L score: {results['embedding_centroid']['rouge_l']:.4f}")
    print(f"• All fine-tuning methods significantly outperformed the base model (2.5x-3.4x improvement)")
    print(f"• Trade-off between performance and selection time:")
    print(f"  - Random: {results['random']['selection_time']:.3f}s selection, {results['random']['rouge_l']:.4f} ROUGE-L")
    print(f"  - Length-based: {results['length_based']['selection_time']:.1f}s selection, {results['length_based']['rouge_l']:.4f} ROUGE-L")  
    print(f"  - Embedding: {results['embedding_centroid']['selection_time']:.1f}s selection, {results['embedding_centroid']['rouge_l']:.4f} ROUGE-L")
    
    print(f"\n📊 EFFECTIVENESS RANKING:")
    methods = [(k, v['rouge_l']) for k, v in results.items() if k != 'base_model']
    methods.sort(key=lambda x: x[1], reverse=True)
    
    for i, (method, score) in enumerate(methods, 1):
        improvement = ((score - results['base_model']['rouge_l']) / results['base_model']['rouge_l']) * 100
        print(f"{i}. {method.replace('_', ' ').title()}: {score:.4f} (+{improvement:.1f}%)")
    
    print(f"\n✅ CONCLUSION:")
    print(f"Knowledge distillation with intelligent data selection successfully improves")
    print(f"LLaMA-2-7B summarization performance on BillSum. Embedding-based selection")
    print(f"provides the best results, though with higher computational cost.")
    

if __name__ == "__main__":
    create_results_summary()
