"""
Evaluation metrics and utilities for tensor completion experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

def compute_metrics(true_values: np.ndarray, 
                   predictions: np.ndarray, 
                   noise_variance: float = 1.0) -> Dict[str, float]:
    """
    Compute evaluation metrics for tensor completion
    
    Parameters:
    -----------
    true_values : np.ndarray
        Ground truth values
    predictions : np.ndarray
        Predicted values
    noise_variance : float
        Noise variance for NLL computation
        
    Returns:
    --------
    metrics : dict
        Dictionary containing RMSE and NLL
    """
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((true_values - predictions)**2))
    
    # Negative Log-Likelihood (assuming Gaussian noise)
    residual = true_values - predictions
    nll = 0.5 * np.log(2 * np.pi * noise_variance) + np.sum(residual**2) / (2 * noise_variance)
    nll = nll / len(residual)  # Average NLL
    
    return {'rmse': rmse, 'nll': nll}


def benchmark_method(method, method_name: str, train_tensor: np.ndarray, 
                    train_mask: np.ndarray, test_values: np.ndarray, 
                    test_indices: np.ndarray, **method_kwargs) -> Dict:
    """
    Benchmark a tensor completion method
    
    Parameters:
    -----------
    method : class
        Method class to benchmark
    method_name : str
        Name of the method for reporting
    train_tensor : np.ndarray
        Training tensor with missing values
    train_mask : np.ndarray
        Training observation mask
    test_values : np.ndarray
        Test values for evaluation
    test_indices : np.ndarray
        Test indices
    **method_kwargs : dict
        Additional keyword arguments for the method
        
    Returns:
    --------
    results : dict
        Benchmark results including metrics and timing
    """
    print(f"\nBenchmarking {method_name}...")
    
    # Initialize method
    model = method(**method_kwargs)
    
    # Time the fitting process
    start_time = time.time()
    model.fit(train_tensor, mask=train_mask)
    fit_time = time.time() - start_time
    
    # Make predictions on test set
    start_time = time.time()
    predictions = model.predict(test_indices)
    predict_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(test_values, predictions)
    
    results = {
        'method': method_name,
        'rmse': metrics['rmse'],
        'nll': metrics['nll'],
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': fit_time + predict_time,
        'convergence_info': getattr(model, 'convergence_info_', {}),
        'model': model  # Store model for analysis
    }
    
    print(f"{method_name} - RMSE: {metrics['rmse']:.6f}, NLL: {metrics['nll']:.6f}, Time: {fit_time:.2f}s")
    
    return results


def run_experiment(tensor: np.ndarray, missing_rates: List[float], 
                  n_runs: int = 5, rank: int = 10, 
                  test_fraction: float = 0.1,
                  standardize: bool = True,
                  results_dir: str = "results") -> pd.DataFrame:
    """
    Run comprehensive tensor completion experiment
    
    Parameters:
    -----------
    tensor : np.ndarray
        Complete tensor for experiments
    missing_rates : List[float]
        List of missing rates to test (e.g., [0.1, 0.25, 0.5])
    n_runs : int
        Number of random runs per missing rate
    rank : int
        CP rank for decomposition
    test_fraction : float
        Fraction of data to use for testing
    standardize : bool
        Whether to standardize the data
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with all experimental results
    """
    from emd_vb_cp import EMDVBCP
    from emd_vb_cp.baselines import CPALS, BayesianCPMCMC
    from utils.data_loader import create_missing_mask, standardize_tensor, split_train_test
    
    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    all_results = []
    
    print(f"Running tensor completion experiment")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Missing rates: {missing_rates}")
    print(f"Number of runs: {n_runs}")
    print(f"CP rank: {rank}")
    
    for missing_rate in missing_rates:
        print(f"\n{'='*50}")
        print(f"Missing rate: {missing_rate*100:.0f}%")
        print(f"{'='*50}")
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}")
            print("-" * 30)
            
            # Create test mask (fixed across methods for fair comparison)
            np.random.seed(42 + run)  # Different seed for each run
            total_entries = np.prod(tensor.shape)
            n_test = int(test_fraction * total_entries)
            test_indices_flat = np.random.choice(total_entries, size=n_test, replace=False)
            test_mask = np.zeros(tensor.shape, dtype=bool)
            test_mask.flat[test_indices_flat] = True
            
            # Create missing mask for training
            train_entries = total_entries - n_test
            n_observed_train = int((1 - missing_rate) * train_entries)
            observed_mask = create_missing_mask(tensor.shape, 1 - missing_rate, random_seed=100 + run)
            
            # Ensure test entries are not in training
            observed_mask = observed_mask & (~test_mask)
            
            # Standardize data
            if standardize:
                from utils.data_loader import standardize_tensor, inverse_standardize_tensor
                tensor_std, std_stats = standardize_tensor(tensor, observed_mask, mode='global')
            else:
                tensor_std = tensor
                std_stats = {}
            
            # Split train/test
            train_tensor, train_mask, test_values, test_indices = split_train_test(
                tensor_std, test_mask, observed_mask)
            
            print(f"Training observations: {np.sum(train_mask)} / {np.prod(tensor.shape)} "
                  f"({100*np.sum(train_mask)/np.prod(tensor.shape):.1f}%)")
            print(f"Test observations: {len(test_values)}")
            
            # Method configurations
            methods = [
                (EMDVBCP, "EMD-VI", {
                    'rank': rank, 
                    'max_iter': 500, 
                    'verbose': False,
                    'tol': 1e-6,
                    'patience': 10
                }),
                (CPALS, "CP-ALS", {
                    'rank': rank, 
                    'max_iter': 500, 
                    'verbose': False,
                    'tol': 1e-6
                }),
                # (BayesianCPMCMC, "Bayesian-CP-MCMC", {
                #     'rank': rank, 
                #     'n_samples': 5000, 
                #     'burn_in': 2000, 
                #     'verbose': False
                # })
            ]
            
            # Benchmark each method
            for method_class, method_name, method_kwargs in methods:
                try:
                    result = benchmark_method(
                        method_class, method_name, train_tensor, train_mask,
                        test_values, test_indices, **method_kwargs
                    )
                    
                    # Add experiment metadata
                    result.update({
                        'missing_rate': missing_rate,
                        'run': run,
                        'rank': rank,
                        'tensor_shape': tensor.shape,
                        'n_train_obs': np.sum(train_mask),
                        'n_test_obs': len(test_values)
                    })
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error running {method_name}: {e}")
                    # Add failed result
                    result = {
                        'method': method_name,
                        'missing_rate': missing_rate,
                        'run': run,
                        'rank': rank,
                        'rmse': np.nan,
                        'nll': np.nan,
                        'fit_time': np.nan,
                        'predict_time': np.nan,
                        'total_time': np.nan,
                        'error': str(e)
                    }
                    all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"tensor_completion_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_df


def plot_results(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot experimental results
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_experiment
    save_path : str, optional
        Path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Group by method and missing rate
    grouped = results_df.groupby(['method', 'missing_rate']).agg({
        'rmse': ['mean', 'std'],
        'nll': ['mean', 'std'],
        'total_time': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns]
    
    missing_rates = sorted(results_df['missing_rate'].unique())
    methods = results_df['method'].unique()
    
    # RMSE plot
    ax = axes[0, 0]
    for method in methods:
        method_data = grouped[grouped['method'] == method]
        ax.errorbar(method_data['missing_rate'], method_data['rmse_mean'], 
                   yerr=method_data['rmse_std'], label=method, marker='o')
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Square Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # NLL plot
    ax = axes[0, 1]
    for method in methods:
        method_data = grouped[grouped['method'] == method]
        ax.errorbar(method_data['missing_rate'], method_data['nll_mean'], 
                   yerr=method_data['nll_std'], label=method, marker='o')
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.set_title('Negative Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Timing plot
    ax = axes[1, 0]
    for method in methods:
        method_data = grouped[grouped['method'] == method]
        ax.errorbar(method_data['missing_rate'], method_data['total_time_mean'], 
                   yerr=method_data['total_time_std'], label=method, marker='o')
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Computation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Performance vs Time scatter
    ax = axes[1, 1]
    for method in methods:
        method_results = results_df[results_df['method'] == method]
        ax.scatter(method_results['total_time'], method_results['rmse'], 
                  label=method, alpha=0.6)
    ax.set_xlabel('Total Time (seconds)')
    ax.set_ylabel('RMSE')
    ax.set_title('Performance vs Time Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
    
    plt.show()


def print_summary_table(results_df: pd.DataFrame):
    """
    Print a summary table of results
    """
    # Calculate summary statistics
    summary = results_df.groupby(['method', 'missing_rate']).agg({
        'rmse': ['mean', 'std'],
        'nll': ['mean', 'std'],
        'total_time': ['mean', 'std']
    }).round(4)
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # Print for each missing rate
    for missing_rate in sorted(results_df['missing_rate'].unique()):
        print(f"\nMissing Rate: {missing_rate*100:.0f}%")
        print("-" * 60)
        print(f"{'Method':<20} {'RMSE':<15} {'NLL':<15} {'Time (s)':<10}")
        print("-" * 60)
        
        for method in results_df['method'].unique():
            method_data = results_df[(results_df['method'] == method) & 
                                   (results_df['missing_rate'] == missing_rate)]
            
            if len(method_data) > 0:
                rmse_mean = method_data['rmse'].mean()
                rmse_std = method_data['rmse'].std()
                nll_mean = method_data['nll'].mean()
                nll_std = method_data['nll'].std()
                time_mean = method_data['total_time'].mean()
                
                print(f"{method:<20} {rmse_mean:.3f}±{rmse_std:.3f:<6} "
                      f"{nll_mean:.3f}±{nll_std:.3f:<6} {time_mean:.1f}")
    
    print("="*80)
