"""
Visualization utilities for knowledge distillation experiments.
Creates quick plots and charts for result analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style

sns.set_palette("husl")

class ResultsVisualizer:
    """Create visualizations for experiment results."""
    
    def __init__(self, output_dir: str = "results/comparison"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_method_comparison(self, results_dict: Dict[str, Dict[str, Any]], 
                             metrics: List[str] = None,
                             save_path: Optional[str] = None) -> None:
        """Create bar chart comparing methods across metrics."""
        if metrics is None:
            metrics = ['rouge1_avg', 'rouge2_avg', 'rougeL_avg']
        
        # Prepare data
        methods = []
        metric_data = {metric: [] for metric in metrics}
        
        for method, data in results_dict.items():
            methods.append(method)
            for metric in metrics:
                score = data['results'].get(metric, 0.0)
                metric_data[metric].append(score)
        
        # Create subplot for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(methods, metric_data[metric])
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_efficiency_frontier(self, results_dict: Dict[str, Dict[str, Any]],
                               metric: str = 'rougeL_avg',
                               save_path: Optional[str] = None) -> None:
        """Plot efficiency frontier (performance vs training data size)."""
        # Prepare data
        methods = []
        scores = []
        train_samples = []
        
        for method, data in results_dict.items():
            methods.append(method)
            scores.append(data['results'].get(metric, 0.0))
            train_samples.append(data.get('train_samples', 0))
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(train_samples, scores, s=100, alpha=0.7)
        
        # Add method labels
        for i, method in enumerate(methods):
            plt.annotate(method, (train_samples[i], scores[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.xlabel('Training Samples')
        plt.ylabel(f'{metric} Score')
        plt.title(f'Efficiency Frontier: {metric} vs Training Data Size')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'efficiency_frontier.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_efficiency(self, results_dict: Dict[str, Dict[str, Any]],
                               metric: str = 'rougeL_avg',
                               save_path: Optional[str] = None) -> None:
        """Plot training efficiency (score per sample)."""
        # Calculate efficiency
        methods = []
        efficiencies = []
        
        for method, data in results_dict.items():
            score = data['results'].get(metric, 0.0)
            samples = data.get('train_samples', 1)
            efficiency = score / samples if samples > 0 else 0
            
            methods.append(method)
            efficiencies.append(efficiency)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, efficiencies)
        plt.ylabel(f'{metric} per Training Sample')
        plt.title('Training Efficiency by Method')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2e}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'training_efficiency.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_method_radar(self, results_dict: Dict[str, Dict[str, Any]],
                         metrics: List[str] = None,
                         save_path: Optional[str] = None) -> None:
        """Create radar chart comparing methods across multiple metrics."""
        if metrics is None:
            metrics = ['rouge1_avg', 'rouge2_avg', 'rougeL_avg']
        
        # Prepare data
        method_names = list(results_dict.keys())
        
        # Set up radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for method in method_names:
            values = []
            for metric in metrics:
                score = results_dict[method]['results'].get(metric, 0.0)
                values.append(score)
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.25)
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title('Method Comparison Radar Chart', y=1.08)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'method_radar.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_table(self, results_dict: Dict[str, Dict[str, Any]],
                           metrics: List[str] = None,
                           save_path: Optional[str] = None) -> pd.DataFrame:
        """Create summary table of all results."""
        if metrics is None:
            metrics = ['rouge1_avg', 'rouge2_avg', 'rougeL_avg']
        
        # Prepare data for DataFrame
        data = []
        for method, result_data in results_dict.items():
            row = {'Method': method}
            
            # Add metric scores
            for metric in metrics:
                score = result_data['results'].get(metric, 0.0)
                row[metric.replace('_avg', '').upper()] = f"{score:.4f}"
            
            # Add additional info
            row['Train Samples'] = result_data.get('train_samples', 'N/A')
            row['Selection Time (s)'] = f"{result_data.get('selection_time', 0):.1f}"
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        if save_path:
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        return df
    
    def generate_full_report(self, results_dict: Dict[str, Dict[str, Any]],
                           metrics: List[str] = None) -> None:
        """Generate comprehensive visual report."""
        logger.info("Generating comprehensive visual report...")
        
        if metrics is None:
            metrics = ['rouge1_avg', 'rouge2_avg', 'rougeL_avg']
        
        # Create all visualizations
        self.plot_method_comparison(results_dict, metrics)
        self.plot_efficiency_frontier(results_dict)
        self.plot_training_efficiency(results_dict)
        
        if len(results_dict) <= 8:  # Radar chart works best with <= 8 methods
            self.plot_method_radar(results_dict, metrics)
        
        # Create summary table
        summary_df = self.create_summary_table(results_dict, metrics)
        
        logger.info(f"Visual report saved to {self.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("VISUAL REPORT SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60)
        
        return summary_df

def quick_plot(results_dict: Dict[str, Dict[str, Any]], 
               plot_type: str = "comparison",
               metric: str = "rougeL_avg") -> None:
    """Quick plotting function for immediate visualization."""
    visualizer = ResultsVisualizer()
    
    if plot_type == "comparison":
        visualizer.plot_method_comparison(results_dict)
    elif plot_type == "efficiency":
        visualizer.plot_efficiency_frontier(results_dict, metric)
    elif plot_type == "radar":
        visualizer.plot_method_radar(results_dict)
    elif plot_type == "all":
        visualizer.generate_full_report(results_dict)
    else:
        logger.warning(f"Unknown plot type: {plot_type}")
        visualizer.plot_method_comparison(results_dict)
