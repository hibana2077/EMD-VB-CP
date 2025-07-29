"""
EMD-VB-CP: Entropy-Regularized Mirror Descent Variational Bayesian CP Decomposition
Core tensor decomposition module implementing the EMD-VI algorithm.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
from typing import Tuple, Optional, List
import warnings

class EMDVBCP:
    """
    Entropy-Regularized Mirror Descent Variational Bayesian CP Decomposition
    
    This class implements the EMD-VI algorithm for low-rank CP tensor decomposition
    with uncertainty quantification through variational inference.
    
    Parameters:
    -----------
    rank : int
        CP rank R for decomposition
    max_iter : int, default=500
        Maximum number of mirror descent iterations
    step_size : float, optional
        Step size for mirror descent (auto-computed if None)
    tol : float, default=1e-6
        Convergence tolerance for ELBO
    patience : int, default=10
        Early stopping patience
    noise_variance : float, default=1.0
        Observation noise variance sigma^2
    kronecker_prior : bool, default=True
        Whether to use Kronecker structured Gaussian prior
    verbose : bool, default=True
        Whether to print progress information
    """
    
    def __init__(self, rank: int, max_iter: int = 500, step_size: Optional[float] = None,
                 tol: float = 1e-6, patience: int = 10, noise_variance: float = 1.0,
                 kronecker_prior: bool = True, verbose: bool = True):
        self.rank = rank
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.patience = patience
        self.noise_variance = noise_variance
        self.kronecker_prior = kronecker_prior
        self.verbose = verbose
        
        # Will be set during fit
        self.shape_ = None
        self.factors_ = None
        self.elbo_history_ = []
        self.convergence_info_ = {}
        
    def _initialize_factors(self, shape: Tuple[int, int, int]) -> List[np.ndarray]:
        """Initialize factor matrices with small random values"""
        factors = []
        for dim in shape:
            # Initialize with small positive values to avoid numerical issues
            factor = np.abs(np.random.randn(dim, self.rank)) + 0.1
            # Normalize each column
            factor = factor / np.linalg.norm(factor, axis=0, keepdims=True)
            factors.append(factor)
        return factors
    
    def _compute_step_size(self, observed_indices: np.ndarray, factors: List[np.ndarray]) -> float:
        """
        Compute adaptive step size using power iteration to estimate Lipschitz constant
        """
        if self.step_size is not None:
            return self.step_size
            
        # Estimate Lipschitz constant via power iteration
        # For CP decomposition, this involves the design matrix structure
        n_obs = len(observed_indices)
        
        # Approximate Lipschitz constant based on problem structure
        # This is a simplified estimation - in practice, more sophisticated methods can be used
        L_approx = max(factor.shape[0] for factor in factors) * self.rank / self.noise_variance
        
        return 0.5 / L_approx
    
    def _kronecker_prior_precision(self, shape: Tuple[int, int, int]) -> List[np.ndarray]:
        """
        Compute Kronecker structured prior precision matrices
        """
        if not self.kronecker_prior:
            # Use identity matrices if not using Kronecker structure
            return [np.eye(dim) for dim in shape]
        
        # Create simple Kronecker structured precision matrices
        # In practice, these could be learned or set based on domain knowledge
        precisions = []
        for dim in shape:
            # Use a simple structure: identity with small regularization
            precision = np.eye(dim) + 0.1 * np.ones((dim, dim)) / dim
            precisions.append(precision)
        return precisions
    
    def _compute_design_matrix_product(self, factors: List[np.ndarray], 
                                     observed_indices: np.ndarray, 
                                     mode: int) -> np.ndarray:
        """
        Compute the design matrix-vector product for mode-n unfolding
        This is the core computational kernel - using Hadamard products for efficiency
        """
        I, J, K = self.shape_
        n_obs = len(observed_indices)
        
        # Get the indices for the current mode
        indices = observed_indices[:, mode]
        
        # Compute Khatri-Rao product of other modes
        other_modes = [m for m in range(3) if m != mode]
        
        # Start with the first "other" mode
        khatri_rao = factors[other_modes[0]][observed_indices[:, other_modes[0]], :]
        
        # Hadamard product with the second "other" mode
        khatri_rao = khatri_rao * factors[other_modes[1]][observed_indices[:, other_modes[1]], :]
        
        # Now compute M_u^T (M_u u - y_u) efficiently
        current_factor = factors[mode]
        
        # Compute M_u u (reconstruction at observed positions)
        reconstruction = np.sum(current_factor[indices, :] * khatri_rao, axis=1)
        
        return khatri_rao, reconstruction, indices
    
    def _mirror_descent_step(self, factor: np.ndarray, gradient: np.ndarray, 
                           step_size: float) -> np.ndarray:
        """
        Perform entropic mirror descent step with closed-form update
        
        u_{t+1} = u_t ⊙ exp(-η ∇f) followed by normalization
        """
        # Entropic mirror step: element-wise exponential
        updated_factor = factor * np.exp(-step_size * gradient)
        
        # Normalize columns to prevent explosion
        norms = np.linalg.norm(updated_factor, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        updated_factor = updated_factor / norms
        
        return updated_factor
    
    def _compute_elbo(self, factors: List[np.ndarray], observed_values: np.ndarray,
                     observed_indices: np.ndarray, precision_matrices: List[np.ndarray]) -> float:
        """
        Compute Evidence Lower Bound (ELBO) for convergence monitoring
        """
        # Reconstruction error term
        reconstruction = self._tensor_reconstruction(factors, observed_indices)
        residual = observed_values - reconstruction
        data_term = -0.5 * np.sum(residual**2) / self.noise_variance
        
        # Prior term (simplified for Kronecker structure)
        prior_term = 0.0
        for i, factor in enumerate(factors):
            # Simplified prior calculation
            factor_norm = np.sum(factor**2)
            prior_term -= 0.5 * factor_norm
        
        return data_term + prior_term
    
    def _tensor_reconstruction(self, factors: List[np.ndarray], 
                             indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct tensor values at specified indices using CP decomposition
        """
        reconstruction = np.zeros(len(indices))
        
        for r in range(self.rank):
            # Compute outer product for rank-1 component
            component = (factors[0][indices[:, 0], r] * 
                        factors[1][indices[:, 1], r] * 
                        factors[2][indices[:, 2], r])
            reconstruction += component
            
        return reconstruction
    
    def fit(self, tensor_data: np.ndarray, mask: Optional[np.ndarray] = None) -> 'EMDVBCP':
        """
        Fit the EMD-VB-CP model to observed tensor data
        
        Parameters:
        -----------
        tensor_data : np.ndarray
            Input tensor of shape (I, J, K)
        mask : np.ndarray, optional
            Boolean mask indicating observed entries (True = observed)
            If None, all entries are considered observed
            
        Returns:
        --------
        self : EMDVBCP
            Fitted model instance
        """
        self.shape_ = tensor_data.shape
        if len(self.shape_) != 3:
            raise ValueError("Input tensor must be 3-dimensional")
        
        # Handle missing values
        if mask is None:
            mask = ~np.isnan(tensor_data)
        
        # Get observed indices and values
        observed_indices = np.column_stack(np.where(mask))
        observed_values = tensor_data[mask]
        
        if self.verbose:
            print(f"Fitting EMD-VB-CP with rank {self.rank}")
            print(f"Tensor shape: {self.shape_}")
            print(f"Observed entries: {len(observed_values)}/{np.prod(self.shape_)} "
                  f"({100*len(observed_values)/np.prod(self.shape_):.1f}%)")
        
        # Initialize factors
        factors = self._initialize_factors(self.shape_)
        
        # Compute step size
        step_size = self._compute_step_size(observed_indices, factors)
        
        # Compute prior precision matrices
        precision_matrices = self._kronecker_prior_precision(self.shape_)
        
        # Main EMD-VI loop
        self.elbo_history_ = []
        best_elbo = -np.inf
        patience_counter = 0
        
        for iteration in range(self.max_iter):
            # Update each factor matrix using mirror descent
            for mode in range(3):
                # Compute design matrix products efficiently
                khatri_rao, reconstruction, indices = self._compute_design_matrix_product(
                    factors, observed_indices, mode)
                
                # Compute gradient
                residual = reconstruction - observed_values
                data_gradient = khatri_rao.T @ residual / self.noise_variance
                
                # Add prior gradient (simplified)
                prior_gradient = factors[mode] * 0.01  # Simple L2 regularization
                
                # Aggregate gradient for each row of the factor matrix
                gradient = np.zeros_like(factors[mode])
                np.add.at(gradient, indices, data_gradient.T)
                gradient += prior_gradient
                
                # Mirror descent step
                factors[mode] = self._mirror_descent_step(factors[mode], gradient, step_size)
            
            # Compute ELBO for convergence monitoring
            if iteration % 10 == 0:  # Compute ELBO every 10 iterations for efficiency
                elbo = self._compute_elbo(factors, observed_values, observed_indices, precision_matrices)
                self.elbo_history_.append(elbo)
                
                # Check convergence
                if len(self.elbo_history_) > 1:
                    elbo_change = abs(self.elbo_history_[-1] - self.elbo_history_[-2])
                    if elbo_change < self.tol:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                        
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Converged at iteration {iteration} (ELBO change: {elbo_change:.2e})")
                        break
                
                if self.verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration}, ELBO: {elbo:.4f}")
        
        # Store final factors and convergence info
        self.factors_ = factors
        self.convergence_info_ = {
            'final_iteration': iteration,
            'final_elbo': self.elbo_history_[-1] if self.elbo_history_ else None,
            'converged': patience_counter >= self.patience
        }
        
        if self.verbose:
            print(f"Training completed. Final ELBO: {self.convergence_info_['final_elbo']:.4f}")
        
        return self
    
    def predict(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict tensor values at specified indices
        
        Parameters:
        -----------
        indices : np.ndarray, optional
            Array of indices where to predict values
            If None, reconstruct the full tensor
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted tensor values
        """
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if indices is None:
            # Reconstruct full tensor
            I, J, K = self.shape_
            predictions = np.zeros(self.shape_)
            
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        for r in range(self.rank):
                            predictions[i, j, k] += (self.factors_[0][i, r] * 
                                                   self.factors_[1][j, r] * 
                                                   self.factors_[2][k, r])
            return predictions
        else:
            # Predict at specific indices
            return self._tensor_reconstruction(self.factors_, indices)
    
    def compute_rmse(self, true_tensor: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Root Mean Square Error on test set
        
        Parameters:
        -----------
        true_tensor : np.ndarray
            Ground truth tensor
        mask : np.ndarray, optional
            Boolean mask for test indices
            
        Returns:
        --------
        rmse : float
            Root Mean Square Error
        """
        if mask is None:
            predictions = self.predict()
            return np.sqrt(np.mean((predictions - true_tensor)**2))
        else:
            test_indices = np.column_stack(np.where(mask))
            predictions = self.predict(test_indices)
            true_values = true_tensor[mask]
            return np.sqrt(np.mean((predictions - true_values)**2))
    
    def compute_negative_log_likelihood(self, true_tensor: np.ndarray, 
                                       mask: Optional[np.ndarray] = None) -> float:
        """
        Compute negative log-likelihood on test set
        
        Parameters:
        -----------
        true_tensor : np.ndarray
            Ground truth tensor
        mask : np.ndarray, optional
            Boolean mask for test indices
            
        Returns:
        --------
        nll : float
            Negative log-likelihood
        """
        if mask is None:
            predictions = self.predict()
            residual = predictions - true_tensor
        else:
            test_indices = np.column_stack(np.where(mask))
            predictions = self.predict(test_indices)
            true_values = true_tensor[mask]
            residual = predictions - true_values
        
        # Gaussian NLL: 0.5 * log(2π σ²) + (residual²) / (2σ²)
        nll = 0.5 * np.log(2 * np.pi * self.noise_variance) + np.sum(residual**2) / (2 * self.noise_variance)
        return nll / len(residual)  # Average NLL
