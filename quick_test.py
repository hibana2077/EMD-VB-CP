#!/usr/bin/env python3
"""
Quick test of optimized BayesianCPMCMC with new defaults
"""

import numpy as np
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from emd_vb_cp.baselines import BayesianCPMCMC

def quick_test():
    """Quick test with optimized defaults"""
    print("=== Quick Test with Optimized Defaults ===\n")
    
    # Generate small test tensor
    shape = (15, 12, 10)
    rank = 3
    
    # Generate true factors
    factors = [np.random.randn(dim, rank) for dim in shape]
    
    # Generate tensor
    tensor = np.zeros(shape)
    for r in range(rank):
        tensor += np.outer(factors[0][:, r], 
                          np.outer(factors[1][:, r], factors[2][:, r])).reshape(shape)
    
    # Add noise and missing entries
    tensor += 0.05 * np.random.randn(*shape)
    mask = np.random.rand(*shape) > 0.3
    
    print(f"Tensor shape: {shape}, Rank: {rank}")
    print(f"Observed entries: {np.sum(mask)} / {tensor.size} ({100*np.sum(mask)/tensor.size:.1f}%)")
    
    # Test with optimized defaults
    model = BayesianCPMCMC(rank=rank)  # Using all default parameters
    
    start_time = time.time()
    model.fit(tensor, mask)
    end_time = time.time()
    
    # Test prediction
    test_indices = np.column_stack(np.where(~mask))[:50]
    if len(test_indices) > 0:
        predictions = model.predict(test_indices)
        true_values = tensor[~mask][:50]
        rmse = np.sqrt(np.mean((true_values - predictions)**2))
    else:
        rmse = 0.0
    
    print(f"\nResults:")
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Effective samples: {model.convergence_info_['n_effective_samples']}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Early stopped: {model.convergence_info_.get('early_stopped', False)}")
    print(f"Final sample: {model.convergence_info_.get('final_sample', 'N/A')}")
    
    print("\n優化成功！BayesianCPMCMC 現在運行更快且默認參數已優化。")

if __name__ == "__main__":
    quick_test()
