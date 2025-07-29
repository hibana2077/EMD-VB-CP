"""
Baseline tensor decomposition methods for comparison
Implements CP-ALS and Bayesian CP-MCMC methods
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import pinv, norm
from sklearn.utils import check_random_state
from typing import Tuple, Optional, List, Dict
import warnings

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
    Simplified implementation for baseline comparison
    """
    
    def __init__(self, rank: int, n_samples: int = 5000, burn_in: int = 2000,
                 noise_precision: float = 1.0, factor_precision: float = 1.0,
                 verbose: bool = True):
        self.rank = rank
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.noise_precision = noise_precision
        self.factor_precision = factor_precision
        self.verbose = verbose
        
        self.factor_samples_ = None
        self.noise_samples_ = None
        self.convergence_info_ = {}
    
    def _sample_factor_conditional(self, factor_idx: int, factors: List[np.ndarray],
                                 observed_data: np.ndarray, observed_indices: np.ndarray,
                                 noise_precision: float) -> np.ndarray:
        """Sample factor matrix from conditional posterior (simplified)"""
        shape = factors[factor_idx].shape
        
        # For simplicity, use a Gibbs-like update with Gaussian approximation
        # This is a simplified version - full implementation would use proper conjugate priors
        
        # Compute sufficient statistics
        other_modes = [i for i in range(3) if i != factor_idx]
        
        # Build design matrix for this mode
        n_obs = len(observed_indices)
        design_matrix = np.ones((n_obs, self.rank))
        
        for i, (idx_i, idx_j, idx_k) in enumerate(observed_indices):
            mode_indices = [idx_i, idx_j, idx_k]
            for r in range(self.rank):
                design_val = 1.0
                for other_mode in other_modes:
                    design_val *= factors[other_mode][mode_indices[other_mode], r]
                design_matrix[i, r] = design_val
        
        # Sample each row of the factor matrix
        new_factor = np.zeros_like(factors[factor_idx])
        
        for row in range(shape[0]):
            # Find observations involving this row
            row_mask = observed_indices[:, factor_idx] == row
            if not np.any(row_mask):
                # No observations for this row, sample from prior
                new_factor[row, :] = np.random.normal(0, 1/np.sqrt(self.factor_precision), self.rank)
                continue
            
            y_row = observed_data[row_mask]
            X_row = design_matrix[row_mask, :]
            
            # Posterior precision and mean
            posterior_precision = self.factor_precision * np.eye(self.rank) + noise_precision * (X_row.T @ X_row)
            posterior_mean = noise_precision * np.linalg.solve(posterior_precision, X_row.T @ y_row)
            
            # Sample from multivariate normal
            try:
                posterior_cov = np.linalg.inv(posterior_precision)
                new_factor[row, :] = np.random.multivariate_normal(posterior_mean, posterior_cov)
            except np.linalg.LinAlgError:
                # Fallback to diagonal approximation
                new_factor[row, :] = np.random.normal(posterior_mean, 1/np.sqrt(np.diag(posterior_precision)))
        
        return new_factor
    
    def _sample_noise_precision(self, factors: List[np.ndarray], observed_data: np.ndarray,
                              observed_indices: np.ndarray) -> float:
        """Sample noise precision from conditional posterior"""
        # Compute residual sum of squares
        reconstruction = np.zeros(len(observed_data))
        for i, (idx_i, idx_j, idx_k) in enumerate(observed_indices):
            for r in range(self.rank):
                reconstruction[i] += (factors[0][idx_i, r] * 
                                    factors[1][idx_j, r] * 
                                    factors[2][idx_k, r])
        
        residual_ss = np.sum((observed_data - reconstruction)**2)
        
        # Gamma prior/posterior (simplified)
        alpha = 1.0 + len(observed_data) / 2
        beta = 1.0 + residual_ss / 2
        
        return np.random.gamma(alpha, 1/beta)
    
    def fit(self, tensor_data: np.ndarray, mask: Optional[np.ndarray] = None) -> 'BayesianCPMCMC':
        """Fit Bayesian CP model using MCMC"""
        shape = tensor_data.shape
        
        if mask is None:
            mask = ~np.isnan(tensor_data)
        
        observed_indices = np.column_stack(np.where(mask))
        observed_data = tensor_data[mask]
        
        if self.verbose:
            print(f"Starting Bayesian CP-MCMC with {self.n_samples} samples ({self.burn_in} burn-in)")
        
        # Initialize factors
        factors = []
        for dim in shape:
            factor = np.random.normal(0, 1, (dim, self.rank))
            factors.append(factor)
        
        noise_precision = self.noise_precision
        
        # Storage for samples
        factor_samples = [[] for _ in range(3)]
        noise_samples = []
        
        # MCMC sampling
        for sample in range(self.n_samples):
            # Sample each factor matrix
            for mode in range(3):
                factors[mode] = self._sample_factor_conditional(
                    mode, factors, observed_data, observed_indices, noise_precision)
            
            # Sample noise precision
            noise_precision = self._sample_noise_precision(factors, observed_data, observed_indices)
            
            # Store samples after burn-in
            if sample >= self.burn_in:
                for mode in range(3):
                    factor_samples[mode].append(factors[mode].copy())
                noise_samples.append(noise_precision)
            
            if self.verbose and sample % 500 == 0:
                print(f"MCMC Sample {sample}/{self.n_samples}")
        
        self.factor_samples_ = factor_samples
        self.noise_samples_ = noise_samples
        self.convergence_info_ = {
            'n_effective_samples': len(noise_samples),
            'mean_noise_precision': np.mean(noise_samples)
        }
        
        if self.verbose:
            print(f"MCMC completed. Effective samples: {len(noise_samples)}")
        
        return self
    
    def predict(self, indices: Optional[np.ndarray] = None, use_mean: bool = True) -> np.ndarray:
        """Predict using posterior mean or samples"""
        if self.factor_samples_ is None:
            raise ValueError("Model has not been fitted yet")
        
        if use_mean:
            # Use posterior mean
            mean_factors = [np.mean(samples, axis=0) for samples in self.factor_samples_]
            
            if indices is None:
                # Reconstruct full tensor
                I, J, K = [f.shape[0] for f in mean_factors]
                tensor = np.zeros((I, J, K))
                for r in range(self.rank):
                    tensor += np.outer(mean_factors[0][:, r], 
                                     np.outer(mean_factors[1][:, r], mean_factors[2][:, r])).reshape(I, J, K)
                return tensor
            else:
                # Predict at specific indices
                predictions = np.zeros(len(indices))
                for i, (idx_i, idx_j, idx_k) in enumerate(indices):
                    for r in range(self.rank):
                        predictions[i] += (mean_factors[0][idx_i, r] * 
                                         mean_factors[1][idx_j, r] * 
                                         mean_factors[2][idx_k, r])
                return predictions
        else:
            # Return prediction uncertainty (not implemented for simplicity)
            raise NotImplementedError("Prediction with uncertainty not implemented")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def compute_nll(y_true: np.ndarray, y_pred: np.ndarray, noise_var: float = 1.0) -> float:
    """Compute Negative Log-Likelihood"""
    residual = y_true - y_pred
    nll = 0.5 * np.log(2 * np.pi * noise_var) + np.sum(residual**2) / (2 * noise_var)
    return nll / len(residual)
