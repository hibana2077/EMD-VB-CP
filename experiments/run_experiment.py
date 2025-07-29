#!/usr/bin/env python3
"""
Main experiment script for EMD-VB-CP tensor completion
Replicates the experiments described in the paper
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import argparse
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emd_vb_cp import EMDVBCP
from emd_vb_cp.baselines import CPALS, BayesianCPMCMC
from utils.data_loader import load_gas_sensor_data
from utils.evaluation import run_experiment, plot_results, print_summary_table

def main():
    parser = argparse.ArgumentParser(description='Run EMD-VB-CP tensor completion experiments')
    parser.add_argument('--data_dir', type=str, default='../src/dataset', 
                       help='Directory containing batch data files')
    parser.add_argument('--results_dir', type=str, default='../results',
                       help='Directory to save results')
    parser.add_argument('--missing_rates', nargs='+', type=float, default=[0.1, 0.25, 0.5],
                       help='Missing rates to test')
    parser.add_argument('--rank', type=int, default=10,
                       help='CP decomposition rank')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of random runs per setting')
    parser.add_argument('--skip_mcmc', action='store_true',
                       help='Skip MCMC baseline (faster)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    print("EMD-VB-CP Tensor Completion Experiment")
    print("="*50)
    
    # Quick test mode for development
    if args.quick_test:
        print("QUICK TEST MODE - Reduced parameters for faster execution")
        args.missing_rates = [0.25]
        args.n_runs = 2
        args.skip_mcmc = True
    
    # Create results directory
    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Missing rates: {args.missing_rates}")
    print(f"CP rank: {args.rank}")
    print(f"Number of runs: {args.n_runs}")
    
    # Load Gas Sensor Array Drift dataset
    print("\nLoading Gas Sensor Array Drift dataset...")
    try:
        tensor, metadata = load_gas_sensor_data(args.data_dir)
        print(f"Loaded tensor shape: {tensor.shape}")
        print(f"Dataset metadata: {metadata}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the dataset is available in the specified directory")
        return 1
    
    # Run experiments
    print("\nStarting tensor completion experiments...")
    
    try:
        results_df = run_experiment(
            tensor=tensor,
            missing_rates=args.missing_rates,
            n_runs=args.n_runs,
            rank=args.rank,
            test_fraction=0.1,
            standardize=True,
            results_dir=args.results_dir
        )
        
        # Print summary
        print_summary_table(results_df)
        
        # Create plots
        plot_path = results_path / "experiment_results.png"
        plot_results(results_df, save_path=str(plot_path))
        
        # Save additional analysis
        analysis_file = results_path / "analysis_summary.txt"
        with open(analysis_file, 'w') as f:
            f.write("EMD-VB-CP Experiment Analysis\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: Gas Sensor Array Drift\n")
            f.write(f"Tensor shape: {tensor.shape}\n")
            f.write(f"Missing rates tested: {args.missing_rates}\n")
            f.write(f"CP rank: {args.rank}\n")
            f.write(f"Number of runs: {args.n_runs}\n\n")
            
            # Performance summary
            best_method_by_rmse = results_df.loc[results_df.groupby('missing_rate')['rmse'].idxmin()]
            f.write("Best method by RMSE for each missing rate:\n")
            for _, row in best_method_by_rmse.iterrows():
                f.write(f"  {row['missing_rate']*100:.0f}% missing: {row['method']} "
                       f"(RMSE: {row['rmse']:.6f})\n")
            
            f.write("\nDetailed Results:\n")
            summary_stats = results_df.groupby(['method', 'missing_rate']).agg({
                'rmse': ['mean', 'std'],
                'nll': ['mean', 'std'], 
                'total_time': ['mean', 'std']
            }).round(6)
            f.write(str(summary_stats))
        
        print(f"\nAnalysis saved to: {analysis_file}")
        print("Experiment completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
