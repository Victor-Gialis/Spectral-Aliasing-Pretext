"""
Example usage of MSE vs F1 analysis tools.

This script demonstrates how to:
1. Load analysis results
2. Extract key metrics
3. Perform custom analysis
4. Generate custom visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import analysis utilities
from utils.analysis_metrics import (
    get_pretrain_metrics,
    get_downstream_metrics,
    analyze_correlation,
    rank_configurations,
    print_analysis_summary,
)

def example_1_basic_loading():
    """Example 1: Load and display basic results."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Loading and Display")
    print("="*70)
    
    csv_path = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        print("Please run: python experiments/analyze_mse_vs_f1.py")
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"\n✅ Loaded {len(df)} configurations")
    print("\nFirst few rows:")
    print(df[['config_id', 'batch_size', 'epochs', 'final_mse', 'f1_score']].head())

def example_2_statistical_summary():
    """Example 2: Generate statistical summary."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Statistical Summary")
    print("="*70)
    
    csv_path = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if not csv_path.exists():
        print(f"❌ Results file not found")
        return
    
    df = pd.read_csv(csv_path)
    
    print_analysis_summary(df)

def example_3_custom_analysis():
    """Example 3: Custom analysis - impact of batch size."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Analysis - Impact of Batch Size")
    print("="*70)
    
    csv_path = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if not csv_path.exists():
        print(f"❌ Results file not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Group by batch size
    batch_analysis = df.groupby('batch_size').agg({
        'final_mse': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max'],
    }).round(4)
    
    print("\nBatch Size Impact:")
    print(batch_analysis)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    batch_groups = df.groupby('batch_size')
    batch_sizes = sorted(df['batch_size'].unique())
    
    mse_means = [batch_groups.get_group(bs)['final_mse'].mean() for bs in batch_sizes]
    f1_means = [batch_groups.get_group(bs)['f1_score'].mean() for bs in batch_sizes]
    
    axes[0].plot(batch_sizes, mse_means, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean MSE', fontsize=11, fontweight='bold')
    axes[0].set_title('MSE vs Batch Size', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(batch_sizes, f1_means, 'o-', color='green', linewidth=2, markersize=8)
    axes[1].set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean F1 Score', fontsize=11, fontweight='bold')
    axes[1].set_title('F1 Score vs Batch Size', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("results/mse_vs_f1_analysis/example_batch_size_impact.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close()

def example_4_find_pareto_optimal():
    """Example 4: Find Pareto-optimal configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Pareto-Optimal Configurations")
    print("="*70)
    
    csv_path = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if not csv_path.exists():
        print(f"❌ Results file not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Pareto optimality: maximize F1, minimize MSE
    # A point is Pareto-optimal if no other point is better in both objectives
    
    pareto_configs = []
    
    for idx1, row1 in df.iterrows():
        is_dominated = False
        
        for idx2, row2 in df.iterrows():
            if idx1 != idx2:
                # row2 dominates row1 if:
                # - row2 has higher F1 AND lower MSE
                if row2['f1_score'] > row1['f1_score'] and row2['final_mse'] < row1['final_mse']:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_configs.append(row1)
    
    print(f"\nFound {len(pareto_configs)} Pareto-optimal configurations:")
    
    pareto_df = pd.DataFrame(pareto_configs)
    print(pareto_df[['config_id', 'batch_size', 'epochs', 'learning_rate', 'final_mse', 'f1_score']].to_string())
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    ax.scatter(df['final_mse'], df['f1_score'], alpha=0.5, s=100, label='All configs', color='lightblue')
    
    # Highlight Pareto-optimal
    ax.scatter(pareto_df['final_mse'], pareto_df['f1_score'], alpha=0.9, s=200, 
               label='Pareto-optimal', color='red', edgecolors='darkred', linewidth=2)
    
    # Connect Pareto points
    pareto_sorted = pareto_df.sort_values('final_mse')
    ax.plot(pareto_sorted['final_mse'], pareto_sorted['f1_score'], 'r--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Final MSE', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Pareto-Optimal Configurations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add config labels
    for idx, row in pareto_df.iterrows():
        ax.annotate(f"C{int(row['config_id'])}", 
                   (row['final_mse'], row['f1_score']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    output_path = Path("results/mse_vs_f1_analysis/example_pareto_optimal.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close()

def example_5_efficiency_score():
    """Example 5: Calculate efficiency score (F1 / computational cost proxy)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Configuration Efficiency Analysis")
    print("="*70)
    
    csv_path = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if not csv_path.exists():
        print(f"❌ Results file not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Proxy for computational cost: batch_size * epochs
    # Higher values = more computation
    df['computational_cost'] = df['batch_size'] * df['epochs']
    
    # Efficiency = F1 / computational cost
    df['efficiency'] = df['f1_score'] / df['computational_cost']
    
    # Efficiency normalized = (F1 + (1-MSE)) / cost
    df['f1_normalized'] = (df['f1_score'] + (1 - df['final_mse'])) / 2
    df['efficiency_normalized'] = df['f1_normalized'] / (df['computational_cost'] / 1000)
    
    efficiency_ranking = df.nlargest(5, 'efficiency_normalized')[
        ['config_id', 'batch_size', 'epochs', 'final_mse', 'f1_score', 
         'computational_cost', 'efficiency_normalized']
    ]
    
    print("\nTop 5 Most Efficient Configurations:")
    print(efficiency_ranking.to_string())

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MSE vs F1 Analysis - Usage Examples")
    print("="*70)
    
    try:
        example_1_basic_loading()
        example_2_statistical_summary()
        example_3_custom_analysis()
        example_4_find_pareto_optimal()
        example_5_efficiency_score()
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
