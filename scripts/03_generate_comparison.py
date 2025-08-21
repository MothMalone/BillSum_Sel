#!/usr/bin/env python3
"""
Generate comprehensive comparison report from experiment results.
Creates visualizations and analysis of all data selection methods.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_results() -> Dict[str, Dict[str, Any]]:
    """Load results from all experiments."""
    results = {}
    results_dir = Path("results")
    
    # Load full baseline if exists
    baseline_path = results_dir / "full_baseline" / "results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            results['full_baseline'] = json.load(f)
        logger.info("Loaded full baseline results")
    
    # Load selection method results
    selection_dir = results_dir / "selection_results"
    if selection_dir.exists():
        for method_dir in selection_dir.iterdir():
            if method_dir.is_dir():
                results_file = method_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results[method_dir.name] = json.load(f)
                    logger.info(f"Loaded {method_dir.name} results")
    
    logger.info(f"Loaded results for {len(results)} methods")
    return results

def generate_comparison_report(results: Dict[str, Dict[str, Any]]):
    """Generate comprehensive comparison report."""
    if not results:
        logger.error("No results found to compare")
        return
    
    # Import visualization tools
    try:
        from utils.visualization import ResultsVisualizer
        from utils.metrics import ComparisonAnalyzer
    except ImportError as e:
        logger.error(f"Failed to import visualization tools: {e}")
        return
    
    # Create output directory
    output_dir = Path("results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tools
    visualizer = ResultsVisualizer(str(output_dir))
    analyzer = ComparisonAnalyzer()
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    try:
        # Method comparison plots
        visualizer.plot_method_comparison(results)
        logger.info("✓ Method comparison plot created")
        
        # Efficiency frontier
        visualizer.plot_efficiency_frontier(results)
        logger.info("✓ Efficiency frontier plot created")
        
        # Training efficiency
        visualizer.plot_training_efficiency(results)
        logger.info("✓ Training efficiency plot created")
        
        # Radar chart (if not too many methods)
        if len(results) <= 8:
            visualizer.plot_method_radar(results)
            logger.info("✓ Radar chart created")
        
        # Summary table
        summary_df = visualizer.create_summary_table(results)
        logger.info("✓ Summary table created")
        
    except Exception as e:
        logger.warning(f"Some visualizations failed: {e}")
    
    # Generate analysis
    logger.info("Generating analysis...")
    
    try:
        # Method comparison analysis
        comparison = analyzer.compare_methods(results, baseline_method='random')
        
        # Save detailed comparison
        with open(output_dir / "detailed_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Method rankings
        rankings = analyzer.rank_methods(results, 'rougeL_avg')
        
        with open(output_dir / "method_rankings.json", 'w') as f:
            json.dump(rankings, f, indent=2)
        
        logger.info("✓ Analysis completed")
        
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
    
    # Generate text report
    generate_text_report(results, rankings if 'rankings' in locals() else None, output_dir)

def generate_text_report(results: Dict[str, Dict[str, Any]], 
                        rankings: list = None,
                        output_dir: Path = None):
    """Generate human-readable text report."""
    if output_dir is None:
        output_dir = Path("results/comparison")
    
    report_lines = [
        "="*80,
        "BILLSUM KNOWLEDGE DISTILLATION FINAL REPORT",
        "="*80,
        f"Generated: {Path.cwd()}",
        f"Total methods evaluated: {len(results)}",
        "",
    ]
    
    # Overall summary
    if 'full_baseline' in results:
        baseline_rouge = results['full_baseline']['results'].get('rougeL_avg', 0)
        report_lines.extend([
            f"FULL DATASET BASELINE:",
            f"  ROUGE-L: {baseline_rouge:.4f}",
            f"  Training samples: {results['full_baseline'].get('train_samples', 'N/A')}",
            "",
        ])
    
    # Method results table
    report_lines.extend([
        "METHOD COMPARISON (10% DATA SELECTION):",
        "-" * 80,
        f"{'Method':<25} {'ROUGE-L':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'Samples':<10}",
        "-" * 80,
    ])
    
    for method, data in results.items():
        if method == 'full_baseline':
            continue
        
        rouge_l = data['results'].get('rougeL_avg', 0)
        rouge_1 = data['results'].get('rouge1_avg', 0)
        rouge_2 = data['results'].get('rouge2_avg', 0)
        samples = data.get('train_samples', 'N/A')
        
        report_lines.append(f"{method:<25} {rouge_l:<10.4f} {rouge_1:<10.4f} {rouge_2:<10.4f} {samples:<10}")
    
    report_lines.extend(["", "-" * 80, ""])
    
    # Rankings
    if rankings:
        report_lines.extend([
            "TOP PERFORMING METHODS:",
            "",
            f"{'Rank':<6} {'Method':<25} {'ROUGE-L':<12} {'Efficiency':<12}",
            "-" * 60,
        ])
        
        for i, ranking in enumerate(rankings[:5]):
            rank = i + 1
            method = ranking['method']
            score = ranking['score']
            efficiency = ranking['efficiency']
            
            report_lines.append(f"{rank:<6} {method:<25} {score:<12.4f} {efficiency:<12.2e}")
        
        report_lines.extend(["", "-" * 60, ""])
    
    # Key insights
    if rankings and len(rankings) > 1:
        best_method = rankings[0]
        best_10p = best_method['method']
        best_score = best_method['score']
        
        if 'full_baseline' in results:
            baseline_score = results['full_baseline']['results'].get('rougeL_avg', 0)
            retention = (best_score / baseline_score) * 100 if baseline_score > 0 else 0
            
            report_lines.extend([
                "KEY INSIGHTS:",
                f"• Best 10% method: {best_10p}",
                f"• Performance retention: {retention:.1f}% of full dataset",
                f"• Data efficiency: 90% reduction in training data",
                f"• Best efficiency: {best_method['efficiency']:.2e} ROUGE-L per sample",
                "",
            ])
    
    # Recommendations
    report_lines.extend([
        "RECOMMENDATIONS:",
        "",
    ])
    
    if rankings:
        top_3 = rankings[:3]
        for i, method_info in enumerate(top_3):
            method = method_info['method']
            report_lines.append(f"{i+1}. {method}: {method_info['score']:.4f} ROUGE-L")
        
        report_lines.extend([
            "",
            f"For maximum efficiency, use '{top_3[0]['method']}' method.",
            "This provides the best performance-to-data ratio for knowledge distillation.",
            "",
        ])
    
    report_lines.extend([
        "="*80,
        "Report generated by BillSum KD Pipeline",
        "="*80
    ])
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open(output_dir / "final_report.txt", 'w') as f:
        f.write(report_text)
    
    # Print to console
    print("\n" + report_text)
    
    logger.info(f"✓ Text report saved to {output_dir / 'final_report.txt'}")

def main():
    """Main function."""
    logger.info("=== GENERATING COMPREHENSIVE COMPARISON REPORT ===")
    
    # Load all results
    results = load_all_results()
    
    if not results:
        logger.error("No experiment results found!")
        logger.error("Make sure to run experiments first:")
        logger.error("  python main.py --mode quick")
        logger.error("  python main.py --mode all")
        return
    
    # Generate report
    generate_comparison_report(results)
    
    logger.info("✅ Comprehensive report generation completed!")
    logger.info("📁 Check results/comparison/ for all outputs")

if __name__ == "__main__":
    main()
