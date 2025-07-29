#!/usr/bin/env python3
"""
Quick demo script to test EMD-VB-CP implementation
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emd_vb_cp import EMDVBCP
from emd_vb_cp.baselines import CPALS
from utils.data_loader import load_gas_sensor_data, create_missing_mask, standardize_tensor
from utils.evaluation import compute_metrics

def main():
    print("EMD-VB-CP Quick Demo")
    print("="*30)
    
    # Load a small subset of data for quick testing
    data_dir = Path(__file__).parent.parent / "src" / "dataset"
    
    try:
        # Load just the first batch for quick demo
        tensor, metadata = load_gas_sensor_data(str(data_dir), batch_files=["batch1.dat", "batch2.dat"])
        print(f"Loaded tensor shape: {tensor.shape}")
        
        # Take a smaller subset for demo
        tensor = tensor[:30, :, :]  # First 30 time points
        print(f"Demo tensor shape: {tensor.shape}")
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Generating synthetic data for demo...")
        
        # Generate synthetic low-rank tensor for demo
        np.random.seed(42)
        I, J, K = 20, 16, 6  # Small tensor for demo
        rank = 3
        
        # Generate factors
        A = np.random.randn(I, rank)
        B = np.random.randn(J, rank) 
        C = np.random.randn(K, rank)
        
        # Create tensor
        tensor = np.zeros((I, J, K))
        for r in range(rank):
            tensor += np.outer(A[:, r], np.outer(B[:, r], C[:, r])).reshape(I, J, K)
        
        # Add noise
        tensor += 0.1 * np.random.randn(I, J, K)
        
        print(f"Generated synthetic tensor shape: {tensor.shape}")
    
    # Create missing values
    missing_rate = 0.3
    mask = create_missing_mask(tensor.shape, missing_rate, random_seed=42)
    
    print(f"Missing rate: {missing_rate*100:.0f}%")
    print(f"Observed entries: {np.sum(mask)} / {np.prod(tensor.shape)}")
    
    # Standardize data
    tensor_std, std_stats = standardize_tensor(tensor, mask, mode='global')
    
    # Create training tensor with missing values
    train_tensor = tensor_std.copy()
    train_tensor[~mask] = np.nan
    
    # Test indices (use a subset of missing entries)
    missing_indices = np.column_stack(np.where(~mask))
    test_size = min(100, len(missing_indices))  # Limit test size for demo
    test_indices = missing_indices[:test_size]
    test_values = tensor_std[~mask][:test_size]
    
    print(f"Test set size: {test_size}")
    
    # Test EMD-VB-CP
    print("\\nTesting EMD-VB-CP...")
    emd_model = EMDVBCP(rank=5, max_iter=100, verbose=True)
    emd_model.fit(train_tensor, mask=mask)
    
    emd_predictions = emd_model.predict(test_indices)
    emd_metrics = compute_metrics(test_values, emd_predictions)
    
    print(f"EMD-VB-CP Results:")
    print(f"  RMSE: {emd_metrics['rmse']:.6f}")
    print(f"  NLL: {emd_metrics['nll']:.6f}")
    
    # Test CP-ALS baseline
    print("\\nTesting CP-ALS baseline...")
    als_model = CPALS(rank=5, max_iter=100, verbose=True)
    als_model.fit(train_tensor, mask=mask)
    
    als_predictions = als_model.predict(test_indices)
    als_metrics = compute_metrics(test_values, als_predictions)
    
    print(f"CP-ALS Results:")
    print(f"  RMSE: {als_metrics['rmse']:.6f}")
    print(f"  NLL: {als_metrics['nll']:.6f}")
    
    # Compare results
    print("\\nComparison:")
    print(f"EMD-VB-CP RMSE improvement: {(als_metrics['rmse'] - emd_metrics['rmse'])/als_metrics['rmse']*100:.1f}%")
    print(f"EMD-VB-CP NLL improvement: {(als_metrics['nll'] - emd_metrics['nll'])/als_metrics['nll']*100:.1f}%")
    
    print("\\nDemo completed successfully!")

if __name__ == "__main__":
    main()
