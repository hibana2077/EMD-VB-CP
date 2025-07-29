#!/usr/bin/env python3
"""
Analyze and visualize tensor completion results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(csv_path):
    """
    Analyze the cleaned CSV results
    """
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Missing rates: {sorted(df['missing_rate'].unique())}")
    print(f"Runs per condition: {df.groupby(['method', 'missing_rate']).size().iloc[0]}")
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    summary_stats = df.groupby(['method', 'missing_rate']).agg({
        'rmse': ['mean', 'std', 'min', 'max'],
        'nll': ['mean', 'std', 'min', 'max'],
        'fit_time': ['mean', 'std'],
        'total_time': ['mean', 'std']
    }).round(4)
    
    print(summary_stats)
    
    # Statistical significance testing
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    from scipy import stats
    
    methods = df['method'].unique()
    missing_rates = sorted(df['missing_rate'].unique())
    
    print("RMSE Comparison (t-test p-values):")
    for missing_rate in missing_rates:
        subset = df[df['missing_rate'] == missing_rate]
        if len(methods) == 2:
            method1_rmse = subset[subset['method'] == methods[0]]['rmse']
            method2_rmse = subset[subset['method'] == methods[1]]['rmse']
            
            statistic, p_value = stats.ttest_ind(method1_rmse, method2_rmse)
            print(f"  Missing rate {missing_rate*100:.0f}%: {methods[0]} vs {methods[1]} - p={p_value:.4f}")
            
            if p_value < 0.05:
                winner = methods[0] if method1_rmse.mean() < method2_rmse.mean() else methods[1]
                print(f"    → {winner} significantly better (p < 0.05)")
            else:
                print(f"    → No significant difference")
    
    # Create visualizations
    create_visualizations(df, csv_path)
    
    return df

def create_visualizations(df, csv_path):
    """
    Create visualizations of the results
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RMSE by missing rate
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='missing_rate', y='rmse', hue='method', ax=ax1)
    ax1.set_title('RMSE by Missing Rate')
    ax1.set_xlabel('Missing Rate')
    ax1.set_ylabel('RMSE')
    
    # 2. NLL by missing rate  
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='missing_rate', y='nll', hue='method', ax=ax2)
    ax2.set_title('Negative Log-Likelihood by Missing Rate')
    ax2.set_xlabel('Missing Rate')
    ax2.set_ylabel('NLL')
    
    # 3. Computation time comparison
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='method', y='fit_time', ax=ax3)
    ax3.set_title('Training Time Comparison')
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. RMSE vs Time scatter
    ax4 = axes[1, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax4.scatter(method_data['fit_time'], method_data['rmse'], 
                   label=method, alpha=0.7, s=50)
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE vs Training Time')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = csv_path.replace('_cleaned.csv', '_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    plt.show()
    
    # Create detailed comparison table
    create_comparison_table(df, csv_path)

def create_comparison_table(df, csv_path):
    """
    Create a detailed comparison table
    """
    print("\n" + "="*60)
    print("DETAILED COMPARISON TABLE")
    print("="*60)
    
    # Pivot table for better readability
    pivot_rmse = df.pivot_table(values='rmse', index='missing_rate', 
                               columns='method', aggfunc=['mean', 'std'])
    pivot_nll = df.pivot_table(values='nll', index='missing_rate', 
                              columns='method', aggfunc=['mean', 'std'])
    pivot_time = df.pivot_table(values='fit_time', index='missing_rate', 
                               columns='method', aggfunc=['mean', 'std'])
    
    print("\nRMSE Results:")
    print(pivot_rmse.round(4))
    
    print("\nNLL Results:")
    print(pivot_nll.round(4))
    
    print("\nTraining Time (seconds):")
    print(pivot_time.round(2))
    
    # Save detailed results
    detailed_path = csv_path.replace('_cleaned.csv', '_summary.txt')
    with open(detailed_path, 'w') as f:
        f.write("TENSOR COMPLETION EXPERIMENT RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Methods: {df['method'].unique()}\n")
        f.write(f"Missing rates: {sorted(df['missing_rate'].unique())}\n")
        f.write(f"Runs per condition: {df.groupby(['method', 'missing_rate']).size().iloc[0]}\n\n")
        
        f.write("RMSE Results (mean ± std):\n")
        f.write(str(pivot_rmse.round(4)) + "\n\n")
        
        f.write("NLL Results (mean ± std):\n")
        f.write(str(pivot_nll.round(4)) + "\n\n")
        
        f.write("Training Time Results (mean ± std seconds):\n")
        f.write(str(pivot_time.round(2)) + "\n\n")
    
    print(f"\nDetailed summary saved to: {detailed_path}")

if __name__ == "__main__":
    # Find cleaned CSV files
    results_dir = Path("results")
    cleaned_files = list(results_dir.glob("*_cleaned.csv"))
    
    if not cleaned_files:
        print("No cleaned CSV files found! Run clean_csv.py first.")
        exit(1)
    
    # Analyze the most recent cleaned file
    latest_file = max(cleaned_files, key=lambda x: x.stat().st_mtime)
    print(f"Analyzing: {latest_file}")
    
    df = analyze_results(str(latest_file))
