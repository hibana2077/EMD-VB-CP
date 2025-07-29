"""
Baseline tensor decomposition methods for comparison
Implements CP-ALS and Bayesian CP-MCMC methods
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import pinv, norm, solve_triangular
from sklearn.utils import check_random_state
from typing import Tuple, Optional, List, Dict
import warnings

# Try to import numba for optimization, fallback if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    
    @jit(nopython=True, parallel=True)
    def _compute_design_matrix_numba(observed_indices, factor_0, factor_1, factor_2, 
                                   factor_idx, rank):
        """Numba-optimized design matrix computation"""
        n_obs = observed_indices.shape[0]
        design_matrix = np.ones((n_obs, rank))
        
        if factor_idx == 0:  # Mode 0, use factors 1 and 2
            for i in prange(n_obs):
                idx_j, idx_k = observed_indices[i, 1], observed_indices[i, 2]
                for r in range(rank):
                    design_matrix[i, r] = factor_1[idx_j, r] * factor_2[idx_k, r]
        elif factor_idx == 1:  # Mode 1, use factors 0 and 2
            for i in prange(n_obs):
                idx_i, idx_k = observed_indices[i, 0], observed_indices[i, 2]
                for r in range(rank):
                    design_matrix[i, r] = factor_0[idx_i, r] * factor_2[idx_k, r]
        else:  # Mode 2, use factors 0 and 1
            for i in prange(n_obs):
                idx_i, idx_j = observed_indices[i, 0], observed_indices[i, 1]
                for r in range(rank):
                    design_matrix[i, r] = factor_0[idx_i, r] * factor_1[idx_j, r]
        
        return design_matrix
    
    @jit(nopython=True, parallel=True)
    def _compute_reconstruction_numba(observed_indices, factor_0, factor_1, factor_2, rank):
        """Numba-optimized reconstruction computation"""
        n_obs = observed_indices.shape[0]
        reconstruction = np.zeros(n_obs)
        
        for i in prange(n_obs):
            idx_i, idx_j, idx_k = observed_indices[i, 0], observed_indices[i, 1], observed_indices[i, 2]
            for r in range(rank):
                reconstruction[i] += factor_0[idx_i, r] * factor_1[idx_j, r] * factor_2[idx_k, r]
        
        return reconstruction
        
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using standard NumPy operations")

class CPALS:
    """
    CP Alternating Least Squares decomposition
    Standard baseline method for tensor completion
    """
    
    def __init__(self, rank: int, max_iter: int = 500, tol: float = 1e-6, 
                 regularization: float = 1e-6, verbose: bool = True):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.verbose = verbose
        
        self.factors_ = None
        self.convergence_info_ = {}
    
    def _initialize_factors(self, shape: Tuple[int, int, int], random_state: int = 42) -> List[np.ndarray]:
        """Initialize factor matrices randomly"""
        rng = check_random_state(random_state)
        factors = []
        for dim in shape:
            factor = rng.randn(dim, self.rank)
            factors.append(factor)
        return factors
    
    def _mode_n_product(self, tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        """Compute mode-n product of tensor with matrix"""
        # Move the mode to the front
        axes = list(range(tensor.ndim))
        axes[0], axes[mode] = axes[mode], axes[0]
        tensor_unfolded = np.transpose(tensor, axes)
        
        # Reshape for matrix multiplication
        original_shape = tensor_unfolded.shape
        tensor_unfolded = tensor_unfolded.reshape(original_shape[0], -1)
        
        # Multiply
        result = matrix @ tensor_unfolded
        
        # Reshape back
        new_shape = (matrix.shape[0],) + original_shape[1:]
        result = result.reshape(new_shape)
        
        # Move the mode back to original position
        result = np.transpose(result, axes)
        return result
    
    def _khatri_rao(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute Khatri-Rao product"""
        return np.concatenate([np.kron(A[:, i:i+1], B[:, i:i+1]) for i in range(A.shape[1])], axis=1)
    
    def _tensor_from_factors(self, factors: List[np.ndarray]) -> np.ndarray:
        """Reconstruct tensor from CP factors"""
        I, J, K = [f.shape[0] for f in factors]
        tensor = np.zeros((I, J, K))
        
        for r in range(self.rank):
            tensor += np.outer(factors[0][:, r], np.outer(factors[1][:, r], factors[2][:, r])).reshape(I, J, K)
        
        return tensor
    
    def fit(self, tensor_data: np.ndarray, mask: Optional[np.ndarray] = None) -> 'CPALS':
        """Fit CP-ALS model"""
        shape = tensor_data.shape
        
        if mask is None:
            mask = ~np.isnan(tensor_data)
        
        # Initialize factors
        factors = self._initialize_factors(shape)
        
        # Handle missing entries by replacing with zeros initially
        tensor_filled = tensor_data.copy()
        tensor_filled[~mask] = 0
        
        reconstruction_errors = []
        
        for iteration in range(self.max_iter):
            old_factors = [f.copy() for f in factors]
            
            # Update each factor
            for mode in range(3):
                # Compute the pseudo-inverse of Khatri-Rao product
                other_modes = [m for m in range(3) if m != mode]
                kr_product = self._khatri_rao(factors[other_modes[0]], factors[other_modes[1]])
                
                # Unfold tensor along mode
                tensor_unfold = np.moveaxis(tensor_filled, mode, 0).reshape(shape[mode], -1)
                
                # Solve least squares with regularization
                A = kr_product.T @ kr_product + self.regularization * np.eye(self.rank)
                b = tensor_unfold @ kr_product
                
                try:
                    factors[mode] = np.linalg.solve(A.T, b.T).T
                except np.linalg.LinAlgError:
                    factors[mode] = pinv(kr_product) @ tensor_unfold.T
                    factors[mode] = factors[mode].T
            
            # Update missing entries with current reconstruction
            current_reconstruction = self._tensor_from_factors(factors)
            tensor_filled[~mask] = current_reconstruction[~mask]
            
            # Check convergence
            factor_change = sum(norm(factors[i] - old_factors[i], 'fro') for i in range(3))
            reconstruction_errors.append(factor_change)
            
            if factor_change < self.tol:
                if self.verbose:
                    print(f"CP-ALS converged at iteration {iteration}")
                break
            
            if self.verbose and iteration % 50 == 0:
                print(f"CP-ALS Iteration {iteration}, factor change: {factor_change:.6f}")
        
        self.factors_ = factors
        self.convergence_info_ = {
            'final_iteration': iteration,
            'reconstruction_errors': reconstruction_errors,
            'converged': factor_change < self.tol
        }
        
        return self
    
    def predict(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict tensor values"""
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet")
        
        if indices is None:
            return self._tensor_from_factors(self.factors_)
        else:
            # Predict at specific indices
            predictions = np.zeros(len(indices))
            for i, (idx_i, idx_j, idx_k) in enumerate(indices):
                for r in range(self.rank):
                    predictions[i] += (self.factors_[0][idx_i, r] * 
                                     self.factors_[1][idx_j, r] * 
                                     self.factors_[2][idx_k, r])
            return predictions


class BayesianCPMCMC:
    """
    Bayesian CP decomposition using MCMC (Gibbs sampling)
    Optimized implementation for better performance
    """
    
    def __init__(self, rank: int, n_samples: int = 2000, burn_in: int = 800,
                 noise_precision: float = 1.0, factor_precision: float = 1.0,
                 verbose: bool = True, thinning: int = 2, 
                 convergence_check: bool = True, convergence_window: int = 200,
                 batch_size: int = None):
        self.rank = rank
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.noise_precision = noise_precision
        self.factor_precision = factor_precision
        self.verbose = verbose
        self.thinning = thinning  # Default to thinning=2 for better performance
        self.convergence_check = convergence_check
        self.convergence_window = convergence_window
        self.batch_size = batch_size  # For batch processing of large datasets
        
        self.factor_samples_ = None
        self.noise_samples_ = None
        self.convergence_info_ = {}
        
        # Pre-allocated arrays for efficiency
        self._design_matrices = None
        self._row_masks = None
    
    def _precompute_structures(self, observed_indices: np.ndarray, shape: Tuple[int, int, int]):
        """Pre-compute data structures for efficient sampling"""
        # Pre-compute row masks for each mode
        self._row_masks = []
        for mode in range(3):
            mode_masks = []
            for row in range(shape[mode]):
                mask = observed_indices[:, mode] == row
                mode_masks.append(mask)
            self._row_masks.append(mode_masks)
    
    def _compute_design_matrix_vectorized(self, factor_idx: int, factors: List[np.ndarray],
                                        observed_indices: np.ndarray) -> np.ndarray:
        """Efficiently compute design matrix using vectorized operations"""
        if NUMBA_AVAILABLE:
            return _compute_design_matrix_numba(
                observed_indices, factors[0], factors[1], factors[2], factor_idx, self.rank)
        
        # Fallback to NumPy implementation
        n_obs = len(observed_indices)
        design_matrix = np.ones((n_obs, self.rank))
        
        other_modes = [i for i in range(3) if i != factor_idx]
        
        # Vectorized computation
        for r in range(self.rank):
            for other_mode in other_modes:
                design_matrix[:, r] *= factors[other_mode][observed_indices[:, other_mode], r]
        
        return design_matrix
    
    def _sample_factor_conditional_vectorized(self, factor_idx: int, factors: List[np.ndarray],
                                            observed_data: np.ndarray, observed_indices: np.ndarray,
                                            noise_precision: float) -> np.ndarray:
        """Optimized factor sampling with vectorized operations and batching"""
        shape = factors[factor_idx].shape
        new_factor = np.zeros_like(factors[factor_idx])
        
        # Pre-compute design matrix once
        design_matrix = self._compute_design_matrix_vectorized(factor_idx, factors, observed_indices)
        
        # Vectorized sampling for all rows with smart batching
        prior_precision_inv = 1.0 / self.factor_precision
        prior_precision = self.factor_precision * np.eye(self.rank)
        
        # Determine batch size for processing rows
        batch_size = self.batch_size or min(50, shape[0])
        
        for batch_start in range(0, shape[0], batch_size):
            batch_end = min(batch_start + batch_size, shape[0])
            batch_rows = range(batch_start, batch_end)
            
            for row in batch_rows:
                row_mask = self._row_masks[factor_idx][row]
                
                if not np.any(row_mask):
                    # Sample from prior
                    new_factor[row, :] = np.random.normal(0, np.sqrt(prior_precision_inv), self.rank)
                    continue
                
                y_row = observed_data[row_mask]
                X_row = design_matrix[row_mask, :]
                
                # Efficient posterior computation using Woodbury matrix identity for small rank
                XTX = X_row.T @ X_row
                XTy = X_row.T @ y_row
                
                if self.rank <= 10:  # Use direct inversion for small rank
                    posterior_precision = prior_precision + noise_precision * XTX
                    
                    try:
                        # Use Cholesky decomposition for numerical stability
                        L = np.linalg.cholesky(posterior_precision)
                        
                        # Solve for posterior mean
                        posterior_mean = solve_triangular(
                            L, solve_triangular(L, noise_precision * XTy, lower=True), 
                            lower=True, trans='T')
                        
                        # Sample using Cholesky factor
                        z = np.random.normal(0, 1, self.rank)
                        sample = posterior_mean + solve_triangular(L, z, lower=True)
                        new_factor[row, :] = sample
                        
                    except np.linalg.LinAlgError:
                        # Fallback to diagonal approximation
                        diag_precision = np.diag(prior_precision) + noise_precision * np.diag(XTX)
                        posterior_mean = noise_precision * XTy / diag_precision
                        posterior_std = 1.0 / np.sqrt(diag_precision)
                        new_factor[row, :] = np.random.normal(posterior_mean, posterior_std)
                
                else:  # Use Woodbury for larger rank
                    # Woodbury matrix identity: (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
                    A_inv = prior_precision_inv * np.eye(self.rank)
                    U = X_row.T
                    V = X_row
                    C_inv = np.eye(len(y_row)) / noise_precision
                    
                    # Compute the Woodbury correction
                    temp = C_inv + V @ A_inv @ U
                    try:
                        temp_inv = np.linalg.inv(temp)
                        posterior_cov = A_inv - A_inv @ U @ temp_inv @ V @ A_inv
                        posterior_mean = posterior_cov @ (noise_precision * XTy)
                        
                        # Sample from multivariate normal
                        new_factor[row, :] = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    except np.linalg.LinAlgError:
                        # Fallback
                        new_factor[row, :] = np.random.normal(0, np.sqrt(prior_precision_inv), self.rank)
        
        return new_factor
    
    def _compute_reconstruction_vectorized(self, factors: List[np.ndarray], observed_indices: np.ndarray) -> np.ndarray:
        """Efficiently compute tensor reconstruction using vectorized operations"""
        if NUMBA_AVAILABLE:
            return _compute_reconstruction_numba(
                observed_indices, factors[0], factors[1], factors[2], self.rank)
        
        # Fallback to NumPy implementation
        # Extract factor values at observed indices
        factor_values = []
        for mode in range(3):
            factor_values.append(factors[mode][observed_indices[:, mode], :])
        
        # Compute reconstruction as element-wise product and sum
        reconstruction = np.sum(factor_values[0] * factor_values[1] * factor_values[2], axis=1)
        return reconstruction
    
    def _sample_noise_precision(self, factors: List[np.ndarray], observed_data: np.ndarray,
                              observed_indices: np.ndarray) -> float:
        """Sample noise precision from conditional posterior - optimized version"""
        # Vectorized reconstruction computation
        reconstruction = self._compute_reconstruction_vectorized(factors, observed_indices)
        residual_ss = np.sum((observed_data - reconstruction)**2)
        
        # Gamma prior/posterior
        alpha = 1.0 + len(observed_data) / 2
        beta = 1.0 + residual_ss / 2
        
        return np.random.gamma(alpha, 1/beta)
    
    def fit(self, tensor_data: np.ndarray, mask: Optional[np.ndarray] = None) -> 'BayesianCPMCMC':
        """Fit Bayesian CP model using optimized MCMC"""
        shape = tensor_data.shape
        
        if mask is None:
            mask = ~np.isnan(tensor_data)
        
        observed_indices = np.column_stack(np.where(mask))
        observed_data = tensor_data[mask]
        
        # Pre-compute efficient data structures
        self._precompute_structures(observed_indices, shape)
        
        if self.verbose:
            print(f"Starting Optimized Bayesian CP-MCMC with {self.n_samples} samples ({self.burn_in} burn-in)")
            print(f"Tensor shape: {shape}, Rank: {self.rank}, Observed entries: {len(observed_data)}")
        
        # Initialize factors with better initialization
        factors = []
        for dim in shape:
            # Use smaller initial variance for better convergence
            factor = np.random.normal(0, 0.1, (dim, self.rank))
            factors.append(factor)
        
        noise_precision = self.noise_precision
        
        # More memory-efficient storage with thinning
        effective_samples = (self.n_samples - self.burn_in) // self.thinning
        factor_samples = [np.zeros((effective_samples, factors[i].shape[0], self.rank)) for i in range(3)]
        noise_samples = np.zeros(effective_samples)
        
        sample_idx = 0
        rmse_history = []
        
        # MCMC sampling with optimizations
        for sample in range(self.n_samples):
            # Sample each factor matrix using optimized method
            for mode in range(3):
                factors[mode] = self._sample_factor_conditional_vectorized(
                    mode, factors, observed_data, observed_indices, noise_precision)
            
            # Sample noise precision
            noise_precision = self._sample_noise_precision(factors, observed_data, observed_indices)
            
            # Monitor convergence
            if sample % 50 == 0:  # Check every 50 samples
                reconstruction = self._compute_reconstruction_vectorized(factors, observed_indices)
                rmse = np.sqrt(np.mean((observed_data - reconstruction)**2))
                rmse_history.append(rmse)
                
                # Early stopping check
                if (self.convergence_check and sample > self.burn_in + self.convergence_window and 
                    len(rmse_history) >= self.convergence_window // 50):
                    recent_rmse = rmse_history[-(self.convergence_window // 50):]
                    if np.std(recent_rmse) < 1e-6:  # Converged
                        if self.verbose:
                            print(f"Early convergence detected at sample {sample}")
                        break
            
            # Store samples after burn-in with thinning
            if sample >= self.burn_in and (sample - self.burn_in) % self.thinning == 0:
                if sample_idx < effective_samples:  # Handle early stopping
                    for mode in range(3):
                        factor_samples[mode][sample_idx] = factors[mode].copy()
                    noise_samples[sample_idx] = noise_precision
                    sample_idx += 1
            
            if self.verbose and sample % max(500, self.n_samples // 20) == 0:
                print(f"MCMC Sample {sample}/{self.n_samples}, RMSE: {rmse_history[-1] if rmse_history else 'N/A':.6f}, Noise precision: {noise_precision:.3f}")
        
        # Trim arrays if early stopping occurred
        if sample_idx < effective_samples:
            for mode in range(3):
                factor_samples[mode] = factor_samples[mode][:sample_idx]
            noise_samples = noise_samples[:sample_idx]
            effective_samples = sample_idx
        
        self.factor_samples_ = factor_samples
        self.noise_samples_ = noise_samples
        self.convergence_info_ = {
            'n_effective_samples': effective_samples,
            'mean_noise_precision': np.mean(noise_samples) if len(noise_samples) > 0 else self.noise_precision,
            'final_rmse': rmse_history[-1] if rmse_history else None,
            'thinning': self.thinning,
            'rmse_history': rmse_history,
            'early_stopped': sample < self.n_samples - 1,
            'final_sample': sample
        }
        
        if self.verbose:
            print(f"MCMC completed. Effective samples: {effective_samples}")
            print(f"Mean noise precision: {np.mean(noise_samples):.3f}")
        
        return self
    
    def predict(self, indices: Optional[np.ndarray] = None, use_mean: bool = True, 
                return_std: bool = False) -> np.ndarray:
        """Predict using posterior mean or samples - optimized version"""
        if self.factor_samples_ is None:
            raise ValueError("Model has not been fitted yet")
        
        if use_mean:
            # Use posterior mean - more efficient computation
            mean_factors = [np.mean(samples, axis=0) for samples in self.factor_samples_]
            
            if indices is None:
                # Reconstruct full tensor using vectorized operations
                I, J, K = [f.shape[0] for f in mean_factors]
                tensor = np.zeros((I, J, K))
                
                # Vectorized reconstruction
                for r in range(self.rank):
                    # Use broadcasting for efficient computation
                    factor_product = np.multiply.outer(mean_factors[0][:, r], 
                                                     np.multiply.outer(mean_factors[1][:, r], 
                                                                     mean_factors[2][:, r]))
                    tensor += factor_product
                
                return tensor
            else:
                # Predict at specific indices using vectorized operations
                predictions = np.zeros(len(indices))
                for r in range(self.rank):
                    predictions += (mean_factors[0][indices[:, 0], r] * 
                                  mean_factors[1][indices[:, 1], r] * 
                                  mean_factors[2][indices[:, 2], r])
                
                if return_std:
                    # Compute prediction standard deviation
                    pred_vars = np.zeros(len(indices))
                    for sample_idx in range(len(self.factor_samples_[0])):
                        sample_pred = np.zeros(len(indices))
                        for r in range(self.rank):
                            sample_pred += (self.factor_samples_[0][sample_idx][indices[:, 0], r] * 
                                          self.factor_samples_[1][sample_idx][indices[:, 1], r] * 
                                          self.factor_samples_[2][sample_idx][indices[:, 2], r])
                        pred_vars += (sample_pred - predictions)**2
                    
                    pred_std = np.sqrt(pred_vars / len(self.factor_samples_[0]))
                    return predictions, pred_std
                
                return predictions
        else:
            # Return prediction uncertainty (simplified implementation)
            raise NotImplementedError("Full posterior sampling not implemented for efficiency")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def compute_nll(y_true: np.ndarray, y_pred: np.ndarray, noise_var: float = 1.0) -> float:
    """Compute Negative Log-Likelihood"""
    residual = y_true - y_pred
    nll = 0.5 * np.log(2 * np.pi * noise_var) + np.sum(residual**2) / (2 * noise_var)
    return nll / len(residual)
