#!/usr/bin/env python3
"""
Script to clean and reformat existing CSV results files
"""
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import sys
import os

def clean_csv_results(csv_path):
    """
    Clean the CSV results file by extracting only the essential information
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read the original CSV
    df = pd.read_csv(csv_path)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Create a cleaned DataFrame
    cleaned_data = []
    
    for idx, row in df.iterrows():
        try:
            # Parse convergence_info if it's a string representation of a dict
            conv_info = {}
            if 'convergence_info' in row and pd.notna(row['convergence_info']):
                try:
                    if isinstance(row['convergence_info'], str):
                        # Try to safely evaluate the string as a dictionary
                        conv_info = ast.literal_eval(row['convergence_info'])
                    elif isinstance(row['convergence_info'], dict):
                        conv_info = row['convergence_info']
                except:
                    print(f"Warning: Could not parse convergence_info for row {idx}")
                    conv_info = {}
            
            # Extract key metrics
            cleaned_row = {
                'method': row.get('method', 'unknown'),
                'rmse': row.get('rmse', np.nan),
                'nll': row.get('nll', np.nan),
                'fit_time': row.get('fit_time', np.nan),
                'predict_time': row.get('predict_time', np.nan),
                'total_time': row.get('total_time', np.nan),
                'missing_rate': row.get('missing_rate', np.nan),
                'run': row.get('run', np.nan),
                'rank': row.get('rank', np.nan),
                'tensor_shape': row.get('tensor_shape', 'unknown'),
                'n_train_obs': row.get('n_train_obs', np.nan),
                'n_test_obs': row.get('n_test_obs', np.nan),
                
                # Extract convergence information
                'final_iteration': conv_info.get('final_iteration', None),
                'converged': conv_info.get('converged', None),
                'final_elbo': conv_info.get('final_elbo', None),
                'n_effective_samples': conv_info.get('n_effective_samples', None),
                'early_stopped': conv_info.get('early_stopped', None)
            }
            
            # Handle reconstruction errors - get the final one if available
            if 'reconstruction_errors' in conv_info and conv_info['reconstruction_errors']:
                errors = conv_info['reconstruction_errors']
                if isinstance(errors, list) and len(errors) > 0:
                    # Get the last reconstruction error
                    cleaned_row['final_reconstruction_error'] = float(errors[-1])
                else:
                    cleaned_row['final_reconstruction_error'] = None
            else:
                cleaned_row['final_reconstruction_error'] = None
            
            cleaned_data.append(cleaned_row)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Create cleaned DataFrame
    cleaned_df = pd.DataFrame(cleaned_data)
    
    print(f"Cleaned CSV shape: {cleaned_df.shape}")
    print(f"Cleaned columns: {list(cleaned_df.columns)}")
    
    # Save cleaned version
    cleaned_path = csv_path.replace('.csv', '_cleaned.csv')
    cleaned_df.to_csv(cleaned_path, index=False)
    
    print(f"Cleaned CSV saved to: {cleaned_path}")
    
    # Display summary
    print("\nData Summary:")
    print(f"Methods: {cleaned_df['method'].unique()}")
    print(f"Missing rates: {sorted(cleaned_df['missing_rate'].unique())}")
    print(f"Number of runs per condition: {cleaned_df.groupby(['method', 'missing_rate']).size().unique()}")
    
    # Display performance summary
    print("\nPerformance Summary:")
    summary = cleaned_df.groupby('method').agg({
        'rmse': ['mean', 'std'],
        'nll': ['mean', 'std'],
        'fit_time': ['mean', 'std'],
        'converged': 'mean'
    }).round(4)
    print(summary)
    
    return cleaned_df

if __name__ == "__main__":
    # Find CSV files in results directory
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("Results directory not found!")
        sys.exit(1)
    
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in results directory!")
        sys.exit(1)
    
    # Clean all CSV files
    for csv_file in csv_files:
        if not csv_file.name.endswith('_cleaned.csv'):  # Skip already cleaned files
            print(f"\n{'='*60}")
            print(f"Cleaning: {csv_file}")
            print(f"{'='*60}")
            try:
                cleaned_df = clean_csv_results(str(csv_file))
            except Exception as e:
                print(f"Error cleaning {csv_file}: {e}")
