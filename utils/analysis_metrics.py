"""
Utilities for extracting and analyzing metrics from MSE vs F1 analysis.

This module provides helper functions to:
1. Extract MSE from pretraining metrics
2. Extract F1 score and other metrics from downstream tasks
3. Build correlation matrices
4. Identify configurations with best/worst performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json

def find_latest_run(base_dir: Path, pattern: str = "metrics.csv") -> Optional[Path]:
    """Find the most recently modified run directory."""
    run_dirs = []
    for d in base_dir.rglob("*"):
        if d.is_dir() and any(pattern in str(f) for f in d.rglob("*")):
            run_dirs.append(d)
    
    if not run_dirs:
        return None
    
    return max(run_dirs, key=lambda p: p.stat().st_mtime)

def get_pretrain_metrics(run_dir: Path) -> Dict[str, float]:
    """
    Extract all pretraining metrics from a run directory.
    
    Returns:
        Dictionary with keys: final_mse, avg_mse, min_mse, max_mse, num_epochs
    """
    metrics_csv = run_dir / "log" / "metrics.csv"
    
    if not metrics_csv.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_csv)
        if df.empty:
            return None
        
        return {
            'final_mse': float(df.iloc[-1]['valid_loss']),
            'avg_mse': float(df['valid_loss'].mean()),
            'min_mse': float(df['valid_loss'].min()),
            'max_mse': float(df['valid_loss'].max()),
            'num_epochs': len(df),
        }
    except Exception as e:
        print(f"Error reading pretrain metrics: {e}")
        return None

def get_downstream_metrics(run_dir: Path) -> Dict[str, float]:
    """
    Extract all downstream metrics from a run directory.
    
    Returns:
        Dictionary with keys: f1_score, accuracy, precision, recall, final_loss
    """
    metrics_csv = run_dir / "log" / "metrics.csv"
    
    if not metrics_csv.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_csv)
        if df.empty:
            return None
        
        metrics = {'f1_score': float(df.iloc[-1].get('f1_score', np.nan))}
        
        # Add other metrics if they exist
        for col in ['accuracy', 'precision', 'recall', 'val_loss']:
            if col in df.columns:
                metrics[col] = float(df.iloc[-1][col])
        
        return metrics
    except Exception as e:
        print(f"Error reading downstream metrics: {e}")
        return None

def load_experiment_config(run_dir: Path) -> Optional[Dict]:
    """Load experiment configuration from config.json."""
    config_path = run_dir / "config.json"
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def analyze_correlation(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute various correlation metrics.
    
    Returns:
        Dictionary with correlation coefficients
    """
    correlations = {
        'pearson': results_df['final_mse'].corr(results_df['f1_score'], method='pearson'),
        'spearman': results_df['final_mse'].corr(results_df['f1_score'], method='spearman'),
        'kendall': results_df['final_mse'].corr(results_df['f1_score'], method='kendall'),
    }
    
    return correlations

def rank_configurations(results_df: pd.DataFrame) -> Dict:
    """Rank configurations by different metrics."""
    return {
        'by_mse': results_df.nsmallest(3, 'final_mse')[['config_id', 'final_mse']].to_dict('records'),
        'by_f1': results_df.nlargest(3, 'f1_score')[['config_id', 'f1_score']].to_dict('records'),
        'by_efficiency': (results_df
                         .assign(efficiency=results_df['f1_score'] / (1 + results_df['final_mse']))
                         .nlargest(3, 'efficiency')[['config_id', 'efficiency']].to_dict('records')),
    }

def print_analysis_summary(results_df: pd.DataFrame):
    """Print a formatted summary of the analysis."""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    correlations = analyze_correlation(results_df)
    rankings = rank_configurations(results_df)
    
    print(f"\nCorrelation Coefficients:")
    print(f"  Pearson:  {correlations['pearson']:.4f}")
    print(f"  Spearman: {correlations['spearman']:.4f}")
    print(f"  Kendall:  {correlations['kendall']:.4f}")
    
    print(f"\nTop 3 Configurations (by MSE - lower is better):")
    for i, row in enumerate(rankings['by_mse'], 1):
        print(f"  {i}. Config {int(row['config_id'])}: MSE = {row['final_mse']:.6f}")
    
    print(f"\nTop 3 Configurations (by F1 Score - higher is better):")
    for i, row in enumerate(rankings['by_f1'], 1):
        print(f"  {i}. Config {int(row['config_id'])}: F1 = {row['f1_score']:.4f}")
    
    print(f"\nTop 3 Most Efficient Configurations (F1 / (1 + MSE)):")
    for i, row in enumerate(rankings['by_efficiency'], 1):
        print(f"  {i}. Config {int(row['config_id'])}: Efficiency = {row['efficiency']:.4f}")

if __name__ == "__main__":
    print(__doc__)
