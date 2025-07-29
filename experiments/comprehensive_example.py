#!/usr/bin/env python3
"""
Comprehensive Example: EMD-VB-CP Tensor Completion
This script demonstrates the complete workflow of using EMD-VB-CP for tensor completion
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emd_vb_cp import EMDVBCP
from emd_vb_cp.baselines import CPALS, BayesianCPMCMC
from utils.data_loader import load_gas_sensor_data, create_missing_mask, standardize_tensor
from utils.evaluation import compute_metrics, run_experiment

def create_synthetic_tensor(I=30, J=16, K=6, rank=3, noise_level=0.1):
    """Create a synthetic low-rank tensor for demonstration"""
    print(f"Creating synthetic tensor of shape ({I}, {J}, {K}) with rank {rank}")
    
    np.random.seed(42)
    
    # Generate factor matrices
    A = np.random.randn(I, rank)
    B = np.random.randn(J, rank)
    C = np.random.randn(K, rank)
    
    # Create tensor from CP decomposition
    tensor = np.zeros((I, J, K))
    for r in range(rank):
        # Outer product of three vectors
        rank_one_tensor = np.outer(A[:, r], np.outer(B[:, r], C[:, r])).reshape(I, J, K)
        tensor += rank_one_tensor
    
    # Add noise
    tensor += noise_level * np.random.randn(I, J, K)
    
    return tensor, (A, B, C)

def visualize_tensor(tensor, title="Tensor Visualization"):
    """Visualize a 3D tensor by showing slices"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for k in range(min(6, tensor.shape[2])):
        ax = axes[k]
        im = ax.imshow(tensor[:, :, k], cmap='viridis', aspect='auto')
        ax.set_title(f'Mode-3 Slice {k+1}')
        ax.set_xlabel('Mode 2 (Sensors)')
        ax.set_ylabel('Mode 1 (Time)')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for k in range(tensor.shape[2], 6):
        axes[k].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def demonstrate_basic_usage():
    """Demonstrate basic EMD-VB-CP usage"""
    print("="*60)
    print("DEMONSTRATION 1: Basic EMD-VB-CP Usage")
    print("="*60)
    
    # Create synthetic data
    tensor, true_factors = create_synthetic_tensor(I=30, J=16, K=6, rank=3)
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Tensor norm: {np.linalg.norm(tensor):.2f}")
    
    # Visualize original tensor
    print("\\nVisualizing original tensor...")
    visualize_tensor(tensor, "Original Tensor")
    
    # Create missing values
    missing_rate = 0.3
    mask = create_missing_mask(tensor.shape, missing_rate, random_seed=42)
    
    print(f"\\nMissing rate: {missing_rate*100:.0f}%")
    print(f"Observed entries: {np.sum(mask)} / {np.prod(tensor.shape)}")
    
    # Standardize data
    tensor_std, std_stats = standardize_tensor(tensor, mask, mode='global')
    print(f"Standardized tensor mean: {np.mean(tensor_std[mask]):.4f}")
    print(f"Standardized tensor std: {np.std(tensor_std[mask]):.4f}")
    
    # Fit EMD-VB-CP
    print("\\nFitting EMD-VB-CP model...")
    start_time = time.time()
    
    model = EMDVBCP(rank=5, max_iter=200, verbose=True, tol=1e-6, patience=10)
    model.fit(tensor_std, mask=mask)
    
    fit_time = time.time() - start_time
    print(f"\\nTraining completed in {fit_time:.2f} seconds")
    
    # Make predictions
    print("\\nMaking predictions...")
    predictions = model.predict()
    
    # Evaluate on missing entries
    test_mask = ~mask
    rmse = model.compute_rmse(tensor_std, mask=test_mask)
    nll = model.compute_negative_log_likelihood(tensor_std, mask=test_mask)
    
    print(f"\\nResults on missing entries:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  NLL: {nll:.6f}")
    
    # Visualize reconstruction
    print("\\nVisualizing reconstruction...")
    visualize_tensor(predictions, "EMD-VB-CP Reconstruction")
    
    # Show convergence
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.elbo_history_)
    plt.title('ELBO Convergence')
    plt.xlabel('Iteration (×10)')
    plt.ylabel('ELBO')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    reconstruction_error = np.abs(predictions - tensor_std)
    plt.hist(reconstruction_error[test_mask], bins=50, alpha=0.7)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, tensor_std, mask

def compare_methods():
    """Compare EMD-VB-CP with baseline methods"""
    print("\\n" + "="*60)
    print("DEMONSTRATION 2: Method Comparison")
    print("="*60)
    
    # Create test data
    tensor, _ = create_synthetic_tensor(I=25, J=16, K=6, rank=4)
    missing_rate = 0.4
    mask = create_missing_mask(tensor.shape, missing_rate, random_seed=123)
    
    # Standardize
    tensor_std, _ = standardize_tensor(tensor, mask)
    
    # Split train/test
    test_mask = ~mask
    test_indices = np.column_stack(np.where(test_mask))[:200]  # Limit for speed
    test_values = tensor_std[test_mask][:200]
    
    print(f"Comparison tensor shape: {tensor.shape}")
    print(f"Missing rate: {missing_rate*100:.0f}%")
    print(f"Test set size: {len(test_values)}")
    
    methods = [
        ("EMD-VB-CP", EMDVBCP(rank=5, max_iter=100, verbose=False)),
        ("CP-ALS", CPALS(rank=5, max_iter=100, verbose=False)),
        ("Bayesian-CP-MCMC", BayesianCPMCMC(rank=5, n_samples=1000, burn_in=200, verbose=False))
    ]
    
    results = []
    
    for method_name, model in methods:
        print(f"\\nTesting {method_name}...")
        
        try:
            start_time = time.time()
            model.fit(tensor_std, mask=mask)
            fit_time = time.time() - start_time
            
            predictions = model.predict(test_indices)
            metrics = compute_metrics(test_values, predictions)
            
            result = {
                'Method': method_name,
                'RMSE': metrics['rmse'],
                'NLL': metrics['nll'],
                'Time (s)': fit_time,
                'Converged': getattr(model, 'convergence_info_', {}).get('converged', 'N/A')
            }
            results.append(result)
            
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  NLL: {metrics['nll']:.6f}")
            print(f"  Time: {fit_time:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            result = {
                'Method': method_name,
                'RMSE': np.nan,
                'NLL': np.nan,
                'Time (s)': np.nan,
                'Converged': False
            }
            results.append(result)
    
    # Display results table
    results_df = pd.DataFrame(results)
    print("\\n" + "="*60)
    print("COMPARISON RESULTS:")
    print("="*60)
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE comparison
    valid_results = results_df.dropna()
    if len(valid_results) > 0:
        ax1.bar(valid_results['Method'], valid_results['RMSE'])
        ax1.set_ylabel('RMSE')
        ax1.set_title('Reconstruction Error Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Time comparison
        ax2.bar(valid_results['Method'], valid_results['Time (s)'])
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Computation Time Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def missing_rate_analysis():
    """Analyze performance across different missing rates"""
    print("\\n" + "="*60)
    print("DEMONSTRATION 3: Missing Rate Analysis")
    print("="*60)
    
    # Create test tensor
    tensor, _ = create_synthetic_tensor(I=20, J=16, K=6, rank=3)
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"Analysis tensor shape: {tensor.shape}")
    print(f"Missing rates to test: {missing_rates}")
    
    results = []
    
    for missing_rate in missing_rates:
        print(f"\\nTesting missing rate: {missing_rate*100:.0f}%")
        
        # Create mask
        mask = create_missing_mask(tensor.shape, missing_rate, random_seed=42)
        
        # Standardize
        tensor_std, _ = standardize_tensor(tensor, mask)
        
        # Test set (subset of missing entries)
        test_mask = ~mask
        n_test = min(100, np.sum(test_mask))  # Limit test size
        test_indices = np.column_stack(np.where(test_mask))[:n_test]
        test_values = tensor_std[test_mask][:n_test]
        
        if len(test_values) == 0:
            print("  No test data available, skipping...")
            continue
        
        # Fit EMD-VB-CP
        try:
            start_time = time.time()
            model = EMDVBCP(rank=4, max_iter=80, verbose=False)
            model.fit(tensor_std, mask=mask)
            fit_time = time.time() - start_time
            
            predictions = model.predict(test_indices)
            metrics = compute_metrics(test_values, predictions)
            
            result = {
                'Missing Rate': missing_rate,
                'Observed %': (1-missing_rate)*100,
                'RMSE': metrics['rmse'],
                'NLL': metrics['nll'],
                'Time (s)': fit_time,
                'ELBO': model.convergence_info_.get('final_elbo', np.nan)
            }
            results.append(result)
            
            print(f"  RMSE: {metrics['rmse']:.6f}, Time: {fit_time:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    if results:
        results_df = pd.DataFrame(results)
        print("\\n" + "="*60)
        print("MISSING RATE ANALYSIS RESULTS:")
        print("="*60)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # RMSE vs observed percentage
        axes[0,0].plot(results_df['Observed %'], results_df['RMSE'], 'o-')
        axes[0,0].set_xlabel('Observed Percentage (%)')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].set_title('RMSE vs Observation Rate')
        axes[0,0].grid(True)
        
        # NLL vs observed percentage
        axes[0,1].plot(results_df['Observed %'], results_df['NLL'], 'o-')
        axes[0,1].set_xlabel('Observed Percentage (%)')
        axes[0,1].set_ylabel('NLL')
        axes[0,1].set_title('NLL vs Observation Rate')
        axes[0,1].grid(True)
        
        # Time vs observed percentage
        axes[1,0].plot(results_df['Observed %'], results_df['Time (s)'], 'o-')
        axes[1,0].set_xlabel('Observed Percentage (%)')
        axes[1,0].set_ylabel('Time (seconds)')
        axes[1,0].set_title('Computation Time vs Observation Rate')
        axes[1,0].grid(True)
        
        # ELBO vs observed percentage
        axes[1,1].plot(results_df['Observed %'], results_df['ELBO'], 'o-')
        axes[1,1].set_xlabel('Observed Percentage (%)')
        axes[1,1].set_ylabel('ELBO')
        axes[1,1].set_title('ELBO vs Observation Rate')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    return None

def real_data_example():
    """Example using real Gas Sensor Array Drift data"""
    print("\\n" + "="*60)
    print("DEMONSTRATION 4: Real Data Example")
    print("="*60)
    
    # Try to load real data
    data_dir = Path(__file__).parent.parent / "src" / "dataset"
    
    try:
        print(f"Loading Gas Sensor Array Drift data from {data_dir}")
        tensor, metadata = load_gas_sensor_data(str(data_dir), 
                                               batch_files=["batch1.dat", "batch2.dat"])
        
        print(f"Loaded real data:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Metadata: {metadata}")
        
        # Take a subset for demonstration
        tensor = tensor[:40, :, :]  # First 40 time points
        print(f"  Demo subset shape: {tensor.shape}")
        
        # Visualize real data
        visualize_tensor(tensor, "Real Gas Sensor Data")
        
        # Apply tensor completion
        missing_rate = 0.25
        mask = create_missing_mask(tensor.shape, missing_rate, random_seed=42)
        
        # Standardize
        tensor_std, stats = standardize_tensor(tensor, mask, mode='sensor')
        
        print(f"\\nApplying tensor completion (25% missing)...")
        model = EMDVBCP(rank=6, max_iter=150, verbose=True)
        model.fit(tensor_std, mask=mask)
        
        # Evaluate
        predictions = model.predict()
        test_mask = ~mask
        rmse = model.compute_rmse(tensor_std, mask=test_mask)
        
        print(f"\\nReal data results:")
        print(f"  RMSE on missing entries: {rmse:.6f}")
        print(f"  Final ELBO: {model.convergence_info_.get('final_elbo', 'N/A')}")
        
        # Visualize reconstruction
        visualize_tensor(predictions, "Reconstructed Gas Sensor Data")
        
        return tensor, predictions, model
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Please ensure the Gas Sensor Array Drift dataset is available")
        print("Skipping real data demonstration...")
        return None, None, None

def main():
    """Run comprehensive demonstration"""
    print("EMD-VB-CP Comprehensive Demonstration")
    print("="*60)
    print("This script demonstrates various aspects of EMD-VB-CP tensor completion")
    print("including basic usage, method comparison, and missing rate analysis.\\n")
    
    # Set up matplotlib for non-interactive use if needed
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend if no display
    except:
        pass
    
    # Run demonstrations
    try:
        # Basic usage
        model, tensor, mask = demonstrate_basic_usage()
        
        # Method comparison
        comparison_results = compare_methods()
        
        # Missing rate analysis
        missing_rate_results = missing_rate_analysis()
        
        # Real data example
        real_tensor, real_predictions, real_model = real_data_example()
        
        print("\\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Key takeaways:")
        print("1. EMD-VB-CP provides fast tensor completion with uncertainty quantification")
        print("2. Performance improves with higher observation rates")
        print("3. Method is suitable for real-world tensor data")
        print("4. Convergence monitoring helps ensure quality results")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
