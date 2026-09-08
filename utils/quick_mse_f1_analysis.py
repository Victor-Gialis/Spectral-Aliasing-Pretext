"""
Quick analysis utility - Extract and display key metrics from MSE vs F1 results.

Usage:
    python utils/quick_mse_f1_analysis.py
"""

import pandas as pd
from pathlib import Path
from tabulate import tabulate  # pip install tabulate

def quick_analysis():
    """Load and display quick analysis summary."""
    
    results_file = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run: python experiments/analyze_mse_vs_f1.py")
        return
    
    df = pd.read_csv(results_file)
    
    if df.empty:
        print("❌ No results found")
        return
    
    print("\n" + "="*80)
    print("MSE vs F1 SCORE - QUICK ANALYSIS")
    print("="*80)
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS")
    print("-" * 80)
    
    summary_data = [
        ["MSE", f"{df['final_mse'].min():.6f}", f"{df['final_mse'].mean():.6f}", f"{df['final_mse'].max():.6f}"],
        ["F1",  f"{df['f1_score'].min():.4f}", f"{df['f1_score'].mean():.4f}", f"{df['f1_score'].max():.4f}"],
    ]
    print(tabulate(summary_data, headers=["Metric", "Min", "Mean", "Max"], tablefmt="grid"))
    
    # Correlation
    correlation = df['final_mse'].corr(df['f1_score'])
    print(f"\n🔗 Correlation (Pearson): {correlation:.4f}")
    
    if correlation < -0.7:
        print("   → STRONG NEGATIVE: Lower MSE strongly associated with higher F1")
    elif correlation < -0.3:
        print("   → MODERATE NEGATIVE: Some relationship between MSE and F1")
    elif correlation > 0.3:
        print("   → WARNING: Unexpected positive correlation")
    else:
        print("   → WEAK: MSE doesn't fully explain F1 variation")
    
    # Top performers
    print("\n🏆 TOP 3 CONFIGURATIONS")
    print("-" * 80)
    
    best_f1 = df.nlargest(3, 'f1_score')[['config_id', 'final_mse', 'f1_score']]
    print("\nBy F1 Score (higher is better):")
    for idx, row in best_f1.iterrows():
        print(f"  Config {int(row['config_id']):2d}: MSE={row['final_mse']:.6f}, F1={row['f1_score']:.4f}")
    
    best_mse = df.nsmallest(3, 'final_mse')[['config_id', 'final_mse', 'f1_score']]
    print("\nBy MSE (lower is better):")
    for idx, row in best_mse.iterrows():
        print(f"  Config {int(row['config_id']):2d}: MSE={row['final_mse']:.6f}, F1={row['f1_score']:.4f}")
    
    # Detailed table
    print("\n📋 DETAILED RESULTS")
    print("-" * 80)
    
    display_df = df[[
        'config_id', 'batch_size', 'learning_rate', 
        'epochs', 'downsampling_factor', 'final_mse', 'f1_score'
    ]].copy()
    
    display_df['final_mse'] = display_df['final_mse'].apply(lambda x: f"{x:.6f}")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}")
    display_df['config_id'] = display_df['config_id'].astype(int)
    display_df['batch_size'] = display_df['batch_size'].astype(int)
    display_df['epochs'] = display_df['epochs'].astype(int)
    display_df['downsampling_factor'] = display_df['downsampling_factor'].astype(int)
    
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    
    # Output file paths
    report_file = Path("results/mse_vs_f1_analysis/analysis_report.txt")
    plots_file = Path("results/mse_vs_f1_analysis/mse_vs_f1_analysis.png")
    
    print("\n📁 OUTPUT FILES")
    print("-" * 80)
    print(f"✓ Results CSV: results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
    
    if plots_file.exists():
        print(f"✓ Main plot: results/mse_vs_f1_analysis/mse_vs_f1_analysis.png")
    else:
        print(f"✗ Main plot: NOT FOUND")
    
    if report_file.exists():
        print(f"✓ Report: results/mse_vs_f1_analysis/analysis_report.txt")
    else:
        print(f"✗ Report: NOT FOUND")
    
    print("\n" + "="*80)
    print("For more details, see: MSE_VS_F1_ANALYSIS_GUIDE.md")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        quick_analysis()
    except ImportError:
        print("Note: For better formatting, install tabulate: pip install tabulate")
        print("Falling back to simple display...\n")
        
        results_file = Path("results/mse_vs_f1_analysis/mse_vs_f1_results.csv")
        if results_file.exists():
            df = pd.read_csv(results_file)
            print(df.to_string())
        else:
            print("❌ Results file not found")
