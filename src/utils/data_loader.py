"""
Data loading and preprocessing utilities for Gas Sensor Array Drift dataset
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import warnings

def load_gas_sensor_data(data_dir: str, batch_files: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Load Gas Sensor Array Drift dataset from batch files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing batch files
    batch_files : List[str], optional
        List of batch filenames to load. If None, loads all batch1.dat to batch10.dat
        
    Returns:
    --------
    tensor : np.ndarray
        3D tensor of shape (T, sensors, gases) where T is time/batch dimension
    metadata : dict
        Metadata about the dataset
    """
    data_path = Path(data_dir)
    
    if batch_files is None:
        batch_files = [f"batch{i}.dat" for i in range(1, 11)]
    
    # Read first file to determine structure
    first_file = data_path / batch_files[0]
    if not first_file.exists():
        raise FileNotFoundError(f"Data file not found: {first_file}")
    
    # Parse first file to understand data format
    with open(first_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Parse the format: label feature1:value1 feature2:value2 ...
    parts = first_line.split()
    gas_label = int(parts[0])  # Gas type (1-6)
    
    # Extract feature indices and values
    features = {}
    for part in parts[1:]:
        if ':' in part:
            idx, val = part.split(':')
            features[int(idx)] = float(val)
    
    # Determine tensor dimensions
    n_features = max(features.keys())  # Should be 128 (16 sensors × 8 features each, but we'll reshape)
    
    # We know from the documentation: 16 sensors, 6 gases, and we need to determine time dimension
    n_sensors = 16
    n_gases = 6
    
    # Load all batch files
    all_data = []
    gas_labels = []
    
    for batch_file in batch_files:
        file_path = data_path / batch_file
        if not file_path.exists():
            warnings.warn(f"Batch file not found: {file_path}, skipping...")
            continue
            
        batch_data = []
        batch_labels = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                gas_label = int(parts[0])
                
                # Extract feature values in order
                features = {}
                for part in parts[1:]:
                    if ':' in part:
                        idx, val = part.split(':')
                        features[int(idx)] = float(val)
                
                # Convert to ordered array (features 1-128)
                feature_vector = np.zeros(128)  # 16 sensors × 8 features per sensor
                for idx, val in features.items():
                    if 1 <= idx <= 128:
                        feature_vector[idx-1] = val  # Convert to 0-based indexing
                
                batch_data.append(feature_vector)
                batch_labels.append(gas_label)
        
        if batch_data:
            all_data.append(np.array(batch_data))
            gas_labels.extend(batch_labels)
    
    if not all_data:
        raise ValueError("No valid data found in batch files")
    
    # Combine all batches - each batch becomes a time slice
    # Result shape: (n_time_points, 128_features)
    combined_data = np.vstack(all_data)
    gas_labels = np.array(gas_labels)
    
    # Reshape to tensor format: (time, sensors, features_per_sensor)
    # We have 128 features = 16 sensors × 8 features per sensor
    n_time_points = combined_data.shape[0]
    features_per_sensor = 128 // n_sensors  # Should be 8
    
    # Reshape to (time, sensors, features_per_sensor)
    tensor_raw = combined_data.reshape(n_time_points, n_sensors, features_per_sensor)
    
    # For the CP decomposition experiment, we need (time, sensors, gases)
    # We'll aggregate over gas types to create a time × sensors × gases tensor
    # This requires grouping measurements by gas type
    
    # Create tensor indexed by gas type
    unique_gases = np.unique(gas_labels)
    n_gases_actual = len(unique_gases)
    
    # Method 1: Average features for each gas type and create time series
    # This creates a meaningful (time, sensors, gases) structure
    
    # Group data by gas type and create time series
    gas_tensors = []
    time_indices = []
    
    for gas in unique_gases:
        gas_mask = gas_labels == gas
        gas_data = tensor_raw[gas_mask]  # Shape: (n_measurements_for_gas, sensors, features)
        
        # Average over features to get (n_measurements, sensors)
        gas_sensor_data = np.mean(gas_data, axis=2)
        gas_tensors.append(gas_sensor_data)
        time_indices.append(np.where(gas_mask)[0])
    
    # Create final tensor by aligning time indices
    max_time = max(len(gt) for gt in gas_tensors)
    final_tensor = np.zeros((max_time, n_sensors, n_gases_actual))
    
    for gas_idx, (gas_data, _) in enumerate(zip(gas_tensors, time_indices)):
        # Pad or truncate to max_time
        if len(gas_data) >= max_time:
            final_tensor[:, :, gas_idx] = gas_data[:max_time, :]
        else:
            final_tensor[:len(gas_data), :, gas_idx] = gas_data
            # Fill remaining with last observation or interpolation
            if len(gas_data) > 0:
                final_tensor[len(gas_data):, :, gas_idx] = gas_data[-1, :]
    
    metadata = {
        'n_time_points': max_time,
        'n_sensors': n_sensors,
        'n_gases': n_gases_actual,
        'gas_labels': unique_gases,
        'original_shape': combined_data.shape,
        'batch_files_loaded': len(all_data),
        'total_measurements': len(combined_data)
    }
    
    return final_tensor, metadata


def create_missing_mask(tensor_shape: Tuple[int, int, int], 
                       missing_rate: float, 
                       random_seed: int = 42) -> np.ndarray:
    """
    Create random missing value mask for tensor completion experiments
    
    Parameters:
    -----------
    tensor_shape : Tuple[int, int, int]
        Shape of the tensor
    missing_rate : float
        Fraction of entries to mask as missing (0.0 to 1.0)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    mask : np.ndarray
        Boolean mask where True indicates observed entries
    """
    np.random.seed(random_seed)
    
    total_entries = np.prod(tensor_shape)
    n_missing = int(missing_rate * total_entries)
    
    # Create mask with all True (observed)
    mask = np.ones(tensor_shape, dtype=bool)
    
    # Randomly select entries to mask
    flat_indices = np.random.choice(total_entries, size=n_missing, replace=False)
    flat_mask = mask.flatten()
    flat_mask[flat_indices] = False
    
    return flat_mask.reshape(tensor_shape)


def standardize_tensor(tensor: np.ndarray, 
                      mask: Optional[np.ndarray] = None, 
                      mode: str = 'global') -> Tuple[np.ndarray, Dict]:
    """
    Standardize tensor using Z-score normalization
    
    Parameters:
    -----------
    tensor : np.ndarray
        Input tensor
    mask : np.ndarray, optional
        Boolean mask indicating observed entries
    mode : str
        Standardization mode: 'global', 'sensor', or 'gas'
        
    Returns:
    --------
    standardized_tensor : np.ndarray
        Standardized tensor
    stats : dict
        Statistics used for standardization (for inverse transform)
    """
    if mask is None:
        mask = ~np.isnan(tensor)
    
    standardized = tensor.copy()
    stats = {}
    
    if mode == 'global':
        # Global standardization
        observed_values = tensor[mask]
        mean_val = np.mean(observed_values)
        std_val = np.std(observed_values)
        
        standardized = (tensor - mean_val) / (std_val + 1e-8)
        stats = {'global_mean': mean_val, 'global_std': std_val}
        
    elif mode == 'sensor':
        # Standardize each sensor independently
        n_sensors = tensor.shape[1]
        sensor_means = np.zeros(n_sensors)
        sensor_stds = np.zeros(n_sensors)
        
        for sensor in range(n_sensors):
            sensor_mask = mask[:, sensor, :]
            if np.any(sensor_mask):
                sensor_values = tensor[:, sensor, :][sensor_mask]
                sensor_mean = np.mean(sensor_values)
                sensor_std = np.std(sensor_values)
                
                standardized[:, sensor, :] = (tensor[:, sensor, :] - sensor_mean) / (sensor_std + 1e-8)
                sensor_means[sensor] = sensor_mean
                sensor_stds[sensor] = sensor_std
        
        stats = {'sensor_means': sensor_means, 'sensor_stds': sensor_stds}
        
    elif mode == 'gas':
        # Standardize each gas independently
        n_gases = tensor.shape[2]
        gas_means = np.zeros(n_gases)
        gas_stds = np.zeros(n_gases)
        
        for gas in range(n_gases):
            gas_mask = mask[:, :, gas]
            if np.any(gas_mask):
                gas_values = tensor[:, :, gas][gas_mask]
                gas_mean = np.mean(gas_values)
                gas_std = np.std(gas_values)
                
                standardized[:, :, gas] = (tensor[:, :, gas] - gas_mean) / (gas_std + 1e-8)
                gas_means[gas] = gas_mean
                gas_stds[gas] = gas_std
        
        stats = {'gas_means': gas_means, 'gas_stds': gas_stds}
    
    else:
        raise ValueError(f"Unknown standardization mode: {mode}")
    
    return standardized, stats


def inverse_standardize_tensor(standardized_tensor: np.ndarray, 
                             stats: Dict, 
                             mode: str = 'global') -> np.ndarray:
    """
    Inverse standardization to recover original scale
    
    Parameters:
    -----------
    standardized_tensor : np.ndarray
        Standardized tensor
    stats : dict
        Statistics from standardization
    mode : str
        Standardization mode used
        
    Returns:
    --------
    tensor : np.ndarray
        Tensor in original scale
    """
    tensor = standardized_tensor.copy()
    
    if mode == 'global':
        mean_val = stats['global_mean']
        std_val = stats['global_std']
        tensor = tensor * std_val + mean_val
        
    elif mode == 'sensor':
        sensor_means = stats['sensor_means']
        sensor_stds = stats['sensor_stds']
        
        for sensor in range(len(sensor_means)):
            tensor[:, sensor, :] = tensor[:, sensor, :] * sensor_stds[sensor] + sensor_means[sensor]
            
    elif mode == 'gas':
        gas_means = stats['gas_means']
        gas_stds = stats['gas_stds']
        
        for gas in range(len(gas_means)):
            tensor[:, :, gas] = tensor[:, :, gas] * gas_stds[gas] + gas_means[gas]
    
    return tensor


def split_train_test(tensor: np.ndarray, 
                    test_mask: np.ndarray,
                    missing_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split tensor into training and test sets for completion experiments
    
    Parameters:
    -----------
    tensor : np.ndarray
        Complete tensor
    test_mask : np.ndarray
        Boolean mask for test set (True = test)
    missing_mask : np.ndarray
        Boolean mask for missing values in training (True = observed)
        
    Returns:
    --------
    train_tensor : np.ndarray
        Training tensor with missing values
    train_mask : np.ndarray
        Training observation mask
    test_tensor : np.ndarray
        Test tensor values
    test_indices : np.ndarray
        Test indices for evaluation
    """
    # Training set: exclude test entries and apply missing mask
    train_mask = missing_mask & (~test_mask)
    train_tensor = tensor.copy()
    train_tensor[~train_mask] = np.nan
    
    # Test set: only test entries
    test_tensor = tensor[test_mask]
    test_indices = np.column_stack(np.where(test_mask))
    
    return train_tensor, train_mask, test_tensor, test_indices
