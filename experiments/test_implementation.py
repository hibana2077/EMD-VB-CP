#!/usr/bin/env python3
"""
Test script to validate EMD-VB-CP implementation
"""

import numpy as np
import sys
from pathlib import Path
import unittest
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emd_vb_cp import EMDVBCP
from emd_vb_cp.baselines import CPALS, BayesianCPMCMC
from utils.data_loader import create_missing_mask, standardize_tensor
from utils.evaluation import compute_metrics

class TestEMDVBCP(unittest.TestCase):
    """Test suite for EMD-VB-CP implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        
        # Create synthetic low-rank tensor
        self.I, self.J, self.K = 10, 8, 6
        self.rank = 3
        
        # Generate factors
        A = np.random.randn(self.I, self.rank)
        B = np.random.randn(self.J, self.rank)
        C = np.random.randn(self.K, self.rank)
        
        # Create tensor from factors
        self.true_tensor = np.zeros((self.I, self.J, self.K))
        for r in range(self.rank):
            self.true_tensor += np.outer(A[:, r], np.outer(B[:, r], C[:, r])).reshape(self.I, self.J, self.K)
        
        # Add small amount of noise
        self.true_tensor += 0.01 * np.random.randn(self.I, self.J, self.K)
        
        # Create missing mask (30% missing)
        self.mask = create_missing_mask(self.true_tensor.shape, 0.3, random_seed=42)
        
        # Standardize
        self.tensor, self.std_stats = standardize_tensor(self.true_tensor, self.mask)
    
    def test_emdvbcp_basic_functionality(self):
        """Test basic EMD-VB-CP functionality"""
        model = EMDVBCP(rank=self.rank, max_iter=50, verbose=False)
        
        # Test fitting
        model.fit(self.tensor, mask=self.mask)
        
        # Check that factors were created
        self.assertIsNotNone(model.factors_)
        self.assertEqual(len(model.factors_), 3)
        
        # Check factor shapes
        self.assertEqual(model.factors_[0].shape, (self.I, self.rank))
        self.assertEqual(model.factors_[1].shape, (self.J, self.rank))
        self.assertEqual(model.factors_[2].shape, (self.K, self.rank))
        
        # Test prediction
        predictions = model.predict()
        self.assertEqual(predictions.shape, self.tensor.shape)
        
        # Test RMSE computation
        rmse = model.compute_rmse(self.tensor, mask=~self.mask)
        self.assertIsInstance(rmse, float)
        self.assertGreater(rmse, 0)
    
    def test_cpals_baseline(self):
        """Test CP-ALS baseline functionality"""
        model = CPALS(rank=self.rank, max_iter=50, verbose=False)
        
        # Test fitting
        model.fit(self.tensor, mask=self.mask)
        
        # Check that factors were created
        self.assertIsNotNone(model.factors_)
        self.assertEqual(len(model.factors_), 3)
        
        # Test prediction
        predictions = model.predict()
        self.assertEqual(predictions.shape, self.tensor.shape)
    
    def test_bayesian_mcmc_baseline(self):
        """Test Bayesian CP-MCMC baseline (reduced parameters for speed)"""
        model = BayesianCPMCMC(rank=self.rank, n_samples=100, burn_in=20, verbose=False)
        
        # Test fitting (with small number of samples for speed)
        model.fit(self.tensor, mask=self.mask)
        
        # Check that samples were created
        self.assertIsNotNone(model.factor_samples_)
        self.assertEqual(len(model.factor_samples_), 3)
        
        # Test prediction
        predictions = model.predict()
        self.assertEqual(predictions.shape, self.tensor.shape)
    
    def test_missing_value_handling(self):
        """Test handling of missing values"""
        # Create tensor with NaN values
        tensor_with_nan = self.tensor.copy()
        tensor_with_nan[~self.mask] = np.nan
        
        model = EMDVBCP(rank=self.rank, max_iter=30, verbose=False)
        
        # Should work with NaN values and no explicit mask
        model.fit(tensor_with_nan)
        predictions = model.predict()
        
        # Predictions should not contain NaN
        self.assertFalse(np.any(np.isnan(predictions)))
    
    def test_convergence_monitoring(self):
        """Test convergence monitoring"""
        model = EMDVBCP(rank=self.rank, max_iter=100, tol=1e-4, patience=5, verbose=False)
        model.fit(self.tensor, mask=self.mask)
        
        # Should have convergence info
        self.assertIsNotNone(model.convergence_info_)
        self.assertIn('final_iteration', model.convergence_info_)
        self.assertIn('converged', model.convergence_info_)
        
        # Should have ELBO history
        self.assertIsInstance(model.elbo_history_, list)
        self.assertGreater(len(model.elbo_history_), 0)
    
    def test_rank_parameter(self):
        """Test different rank parameters"""
        for rank in [1, 3, 5]:
            model = EMDVBCP(rank=rank, max_iter=20, verbose=False)
            model.fit(self.tensor, mask=self.mask)
            
            # Check factor dimensions
            for factor in model.factors_:
                self.assertEqual(factor.shape[1], rank)
    
    def test_prediction_at_indices(self):
        """Test prediction at specific indices"""
        model = EMDVBCP(rank=self.rank, max_iter=30, verbose=False)
        model.fit(self.tensor, mask=self.mask)
        
        # Test prediction at specific indices
        test_indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        predictions = model.predict(test_indices)
        
        self.assertEqual(len(predictions), len(test_indices))
        self.assertFalse(np.any(np.isnan(predictions)))
    
    def test_metrics_computation(self):
        """Test metrics computation functions"""
        # Generate some test predictions
        true_values = np.random.randn(100)
        predictions = true_values + 0.1 * np.random.randn(100)
        
        metrics = compute_metrics(true_values, predictions)
        
        self.assertIn('rmse', metrics)
        self.assertIn('nll', metrics)
        self.assertIsInstance(metrics['rmse'], float)
        self.assertIsInstance(metrics['nll'], float)
        self.assertGreater(metrics['rmse'], 0)
    
    def test_step_size_computation(self):
        """Test automatic step size computation"""
        model = EMDVBCP(rank=self.rank, step_size=None, max_iter=10, verbose=False)
        model.fit(self.tensor, mask=self.mask)
        
        # Should have computed step size automatically
        # (This is tested indirectly through successful fitting)
        self.assertIsNotNone(model.factors_)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        model = EMDVBCP(rank=self.rank)
        
        # Test with wrong tensor dimension
        with self.assertRaises(ValueError):
            model.fit(np.random.randn(10, 10))  # 2D instead of 3D
        
        # Test prediction before fitting
        unfitted_model = EMDVBCP(rank=self.rank)
        with self.assertRaises(ValueError):
            unfitted_model.predict()


def run_performance_test():
    """Run a performance test to check timing"""
    print("\nRunning performance test...")
    
    # Create larger tensor for performance testing
    np.random.seed(42)
    I, J, K = 50, 16, 6
    rank = 3
    
    # Generate factors
    A = np.random.randn(I, rank)
    B = np.random.randn(J, rank)
    C = np.random.randn(K, rank)
    
    # Create tensor
    tensor = np.zeros((I, J, K))
    for r in range(rank):
        tensor += np.outer(A[:, r], np.outer(B[:, r], C[:, r])).reshape(I, J, K)
    tensor += 0.1 * np.random.randn(I, J, K)
    
    # Create missing mask
    mask = create_missing_mask(tensor.shape, 0.4, random_seed=42)
    
    # Standardize
    tensor_std, _ = standardize_tensor(tensor, mask)
    
    print(f"Performance test tensor shape: {tensor.shape}")
    print(f"Observed entries: {np.sum(mask)} / {np.prod(tensor.shape)} ({100*np.sum(mask)/np.prod(tensor.shape):.1f}%)")
    
    # Test EMD-VB-CP
    import time
    
    print("\nTesting EMD-VB-CP performance...")
    start_time = time.time()
    model = EMDVBCP(rank=5, max_iter=100, verbose=False)
    model.fit(tensor_std, mask=mask)
    end_time = time.time()
    
    print(f"EMD-VB-CP fitting time: {end_time - start_time:.2f} seconds")
    print(f"Final ELBO: {model.convergence_info_.get('final_elbo', 'N/A')}")
    print(f"Converged: {model.convergence_info_.get('converged', 'N/A')}")
    
    # Test reconstruction quality
    test_mask = ~mask
    if np.any(test_mask):
        test_indices = np.column_stack(np.where(test_mask))[:100]  # Limit for speed
        test_values = tensor_std[test_mask][:100]
        
        predictions = model.predict(test_indices)
        metrics = compute_metrics(test_values, predictions)
        
        print(f"Test RMSE: {metrics['rmse']:.6f}")
        print(f"Test NLL: {metrics['nll']:.6f}")


def main():
    """Run all tests"""
    print("EMD-VB-CP Test Suite")
    print("=" * 40)
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    run_performance_test()
    
    print("\n" + "=" * 40)
    print("All tests completed!")


if __name__ == "__main__":
    main()
