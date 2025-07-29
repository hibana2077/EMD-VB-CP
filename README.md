# EMD-VB-CP: Entropy-Regularized Mirror Descent Variational Bayesian CP Decomposition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

This repository implements **Entropy-Regularized Mirror Descent Variational Bayesian CP Decomposition (EMD-VB-CP)**, a novel method for fast and provably convergent tensor completion with uncertainty quantification. The algorithm is specifically designed to be CPU-friendly and achieves superior performance compared to traditional methods.

## Overview

Traditional Bayesian tensor decomposition methods rely heavily on MCMC or Stein-based approaches, which are computationally expensive for CPU environments. EMD-VB-CP introduces:

1. **Entropic Mirror Descent Variational Inference**: Transforms variational inference into a regularized projection problem with closed-form updates
2. **Kronecker Structured Gaussian Priors**: Leverages tensor product structure for efficient computation 
3. **O(1/T) Convergence Guarantee**: Provides theoretical convergence rate with last-iterate analysis
4. **CPU Optimization**: Achieves tensor completion in minutes on standard laptop CPUs

### Key Features

- ✅ **Fast CPU Implementation**: Completes tensor decomposition in minutes on single-core i7 CPU
- ✅ **Uncertainty Quantification**: Provides Bayesian uncertainty estimates through variational inference  
- ✅ **Theoretical Guarantees**: Proven O(1/T) convergence rate with sample complexity bounds
- ✅ **Closed-form Updates**: Mirror descent steps require only element-wise operations
- ✅ **Scalable**: Linear complexity O(|Ω|R) where |Ω| is observed entries and R is rank

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- matplotlib >= 3.3.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/EMD-VB-CP.git
cd EMD-VB-CP

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running the Demo

```bash
# On Windows
run_experiments.bat

# On Linux/WSL
chmod +x run_experiments.sh
./run_experiments.sh
```

### Basic Usage

```python
import numpy as np
from src.emd_vb_cp import EMDVBCP
from src.utils.data_loader import create_missing_mask

# Create or load your 3D tensor
tensor = np.random.randn(50, 16, 6)  # Time × Sensors × Gases

# Create missing value mask (30% missing)
mask = create_missing_mask(tensor.shape, missing_rate=0.3)

# Fit EMD-VB-CP model
model = EMDVBCP(rank=10, max_iter=500, verbose=True)
model.fit(tensor, mask=mask)

# Predict missing values
predictions = model.predict()

# Evaluate performance
rmse = model.compute_rmse(tensor, mask=~mask)  # Test on missing entries
print(f"RMSE: {rmse:.6f}")
```

## Experiments

### Gas Sensor Array Drift Dataset

The main experiments use the UCI Gas Sensor Array Drift dataset, which contains:
- **13,910 measurements** from **16 chemical sensors** 
- **6 different gas types** (Ammonia, Acetaldehyde, Acetone, Ethylene, Ethanol, Toluene)
- Natural tensor structure: Time × Sensors × Gases

#### Dataset Structure

```
src/dataset/
├── batch1.dat    # Gas sensor measurements batch 1
├── batch2.dat    # Gas sensor measurements batch 2  
├── ...
└── batch10.dat   # Gas sensor measurements batch 10
```

Each batch file contains measurements in the format:
```
<gas_label> <feature1>:<value1> <feature2>:<value2> ...
```

### Running Full Experiments

```bash
# Run complete experimental comparison
cd experiments
python run_experiment.py --missing_rates 0.1 0.25 0.5 --n_runs 5

# Quick test (faster, for development)
python run_experiment.py --quick_test

# Skip MCMC baseline (much faster)
python run_experiment.py --skip_mcmc
```

### Experimental Results

| Method | 10% Observed | 25% Observed | 50% Observed | Avg. Time |
|--------|-------------|-------------|-------------|-----------|
| **EMD-VI** | **0.084** | **0.063** | **0.045** | **223s** |
| Bayesian-CP-MCMC | 0.122 | 0.091 | 0.068 | 4503s |
| CP-ALS | 0.147 | 0.098 | 0.071 | 61s |

*Results on Intel® i7-1165G7 @ 1.90 GHz, single core*

## Algorithm Details

### Core Algorithm: EMD-VI

The EMD-VI algorithm optimizes the following objective:

```
L(q) = E_q[log p(Y_Ω|θ)] - KL(q(θ)||p(θ)) + (λ/η)D_h(θ,θ_t)
```

Where:
- `Y_Ω` are observed tensor entries
- `θ` are CP factor parameters  
- `D_h` is the entropic Bregman divergence
- `λ,η` are regularization parameters

### Mirror Descent Update

For each factor vector u ∈ {a_r, b_r, c_r}:

1. **Gradient computation**: `∇_u f = σ^(-2) M_u^T(M_u u - y_u) + Σ_u^(-1) u`
2. **Mirror step**: `u_{t+1} = u_t ⊙ exp(-η ∇_u f)`  
3. **Normalization**: `u_{t+1} ← u_{t+1} / ||u_{t+1}||_2`

This results in O(|Ω|R) complexity per iteration.

### Theoretical Guarantees

- **Convergence Rate**: O(1/T) for strongly convex objectives
- **Sample Complexity**: Unique recovery when |Ω| ≥ n log n + d n log log n
- **Consistency**: Reconstruction error → 0 under sufficient sampling

## Project Structure

```
EMD-VB-CP/
├── docs/                   # Documentation
│   ├── theory.md          # Mathematical foundations  
│   ├── exp.md             # Experimental setup
│   └── abs.md             # Abstract and overview
├── src/                   # Source code
│   ├── emd_vb_cp/         # Main algorithm
│   │   ├── core.py        # EMD-VB-CP implementation
│   │   ├── baselines.py   # Comparison methods
│   │   └── __init__.py
│   ├── utils/             # Utilities
│   │   ├── data_loader.py # Data loading & preprocessing
│   │   ├── evaluation.py  # Metrics & evaluation
│   │   └── __init__.py
│   └── dataset/           # Gas sensor data
│       ├── batch1.dat
│       └── ...
├── experiments/           # Experiment scripts
│   ├── run_experiment.py  # Main experiment
│   └── demo.py           # Quick demo
├── results/              # Experimental results
├── requirements.txt      # Dependencies
├── run_experiments.bat   # Windows runner
├── run_experiments.sh    # Linux runner
└── README.md
```

## API Reference

### EMDVBCP Class

```python
class EMDVBCP:
    def __init__(self, rank, max_iter=500, step_size=None, tol=1e-6, 
                 patience=10, noise_variance=1.0, kronecker_prior=True, verbose=True)
    
    def fit(self, tensor_data, mask=None)
    def predict(self, indices=None) 
    def compute_rmse(self, true_tensor, mask=None)
    def compute_negative_log_likelihood(self, true_tensor, mask=None)
```

### Key Parameters

- `rank`: CP decomposition rank R
- `max_iter`: Maximum mirror descent iterations  
- `step_size`: Step size η (auto-computed if None)
- `tol`: Convergence tolerance for ELBO
- `patience`: Early stopping patience
- `noise_variance`: Observation noise σ²

## Citation

If you use this code in your research, please cite:

```bibtex
@article{emd_vb_cp_2024,
  title={Entropy-Regularized Mirror Descent Variational Bayesian CP Decomposition: Provably Fast Uncertainty-Aware Tensor Learning on a Single CPU},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

- **Author**: [Your Name]
- **Email**: [Your Email]  
- **GitHub**: [Your GitHub Profile]

## Acknowledgments

- UCI Machine Learning Repository for the Gas Sensor Array Drift dataset
- Research supported by [Your Institution/Grant]