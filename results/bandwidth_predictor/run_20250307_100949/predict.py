
import torch
import pickle
import json
import pandas as pd
import numpy as np

def predict_bandwidth(config, model_info):
    """
    Predict communication bandwidth for a given model and configuration.

    Args:
        config: Dict with runtime config (workers, batch_size, ps, tensorsize)
        model_info: Dict with model architecture info (num_parameters, num_layers, etc.)

    Returns:
        Predicted bandwidth in MB/s
    """
    # Load model
    model_path = "best_model.pth"
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # Load scaler
    with open("feature_scaler.pkl", 'rb') as f:
        feature_scaler = pickle.load(f)

    # Load feature names
    with open("feature_names.json", 'r') as f:
        feature_names = json.load(f)

    # Create a dataframe with the new configuration
    new_data = {
        'tensorsize': config.get('tensorsize', 1),
        'Number of Workers': config.get('num_workers', 1),
        'Batch Size': config.get('batch_size', 32),
        'Number of PSs': config.get('num_ps', 0),
        'Data Set': config.get('dataset', 'cifar10'),
        'Pattern': config.get('pattern', 'allreduce').lower()  # Ensure lowercase
    }

    # Add model architecture information (only include what was in training)
    new_data.update({
        'Number of Parameters': model_info.get('num_parameters', 0),
        'Number of Layers': model_info.get('num_layers', 0)
    })

    # Convert to DataFrame
    df = pd.DataFrame([new_data])

    # Extract features
    features = extract_model_independent_features(df)

    # Ensure features match those used during training
    scaler_feature_names = feature_scaler.feature_names_in_

    # Add missing columns with zeros
    for col in scaler_feature_names:
        if col not in features.columns:
            features[col] = 0

    # Keep only columns that were present during training
    features = features[scaler_feature_names]

    # Scale features
    scaled_features = feature_scaler.transform(features)

    # Convert to tensor
    input_tensor = torch.FloatTensor(scaled_features)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        # Convert back from log scale
        predicted_bandwidth = torch.exp(output).item() - 1

    return predicted_bandwidth

def extract_model_independent_features(df):
    """Create model-independent features that can be used for any model"""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Create feature DataFrame
    features = pd.DataFrame()

    # Basic features
    features['tensor_size'] = df['tensorsize'].astype(float)
    features['num_workers'] = df['Number of Workers'].astype(float)
    features['batch_size'] = df['Batch Size'].astype(float)
    features['num_ps'] = df['Number of PSs'].astype(float)

    # Communication pattern features
    N = df['Number of Workers']
    features['is_ring_allreduce'] = (df['Number of PSs'] == 0).astype(float)

    # Ring AllReduce features
    features['ring_volume'] = df['tensorsize'] * 2 * (N-1) / N * features['is_ring_allreduce']
    features['ring_steps'] = 2 * (N-1) * features['is_ring_allreduce']

    # PS features
    ps_mask = 1 - features['is_ring_allreduce']
    features['ps_volume'] = df['tensorsize'] * N * 2 * ps_mask
    features['ps_worker_ratio'] = (N / df['Number of PSs'].clip(lower=1)) * ps_mask

    # Model architecture features (if available)
    if 'Number of Parameters' in df.columns:
        features['num_parameters'] = df['Number of Parameters'].astype(float)
    if 'Number of Layers' in df.columns:
        features['num_layers'] = df['Number of Layers'].astype(float)
    # Intentionally not including model_depth

    # Communication/computation features
    features['comm_intensity'] = features['tensor_size'] * N / features['batch_size']

    # Log-transformed features
    features['log_tensor_size'] = np.log1p(df['tensorsize'])
    features['log_batch_size'] = np.log(df['Batch Size'])
    features['log_workers'] = np.log2(N)

    # Dataset size (if available)
    if 'Data Set' in df.columns:
        features['dataset_size'] = df['Data Set'].map({
            'cifar10': 1,
            'imagenet': 10,
            'squad': 5
        }).fillna(1)  # Default to 1 for unknown datasets

    # Pattern one-hot encoding (if available)
    if 'Pattern' in df.columns:
        # Convert pattern to lowercase for consistency
        lowercase_pattern = df['Pattern'].str.lower()
        pattern_dummies = pd.get_dummies(lowercase_pattern, prefix='pattern')
        features = pd.concat([features, pattern_dummies], axis=1)

    return features

# Example usage
if __name__ == "__main__":
    config = {
        'tensorsize': 10000000,  # 10MB tensor
        'num_workers': 4,
        'batch_size': 64,
        'num_ps': 0,  # AllReduce
        'dataset': 'imagenet',
        'pattern': 'allreduce'  # Lowercase for consistency
    }

    model_info = {
        'num_parameters': 25000000,  # 25M params
        'num_layers': 50
        # No model_depth
    }

    bw = predict_bandwidth(config, model_info)
    print(f"Predicted bandwidth: {bw:.2f} MB/s")
    