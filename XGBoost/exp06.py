'''
모델명 One-hot encoding 제외함.

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import time
import os
import pickle
import json

class CommDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class BandwidthPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Main network with consistent input dimension regardless of model
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def extract_model_independent_features(df):
    """
    Create model-independent features that can be used for any model
    """
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
    # Note: model_depth is intentionally not included if it wasn't in original training

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

def train_model(train_loader, val_loader, input_dim, device, output_dir, model_name="MLP"):
    """Train and save a model for bandwidth prediction"""
    # Create model
    model = BandwidthPredictor(input_dim).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Training loop
    num_epochs = 300
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Ensure targets shape matches outputs
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.view(-1, 1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(output_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(output_dir, f"{model_name}_training_curve.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Model saved to {model_path}")
    print(f"Training curve saved to {plot_path}")

    # Load best model for return
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Convert back from log scale
            pred_exp = torch.exp(outputs) - 1
            actual_exp = torch.exp(targets.view(-1, 1)) - 1

            all_predictions.extend(pred_exp.cpu().numpy().flatten())
            all_targets.extend(actual_exp.cpu().numpy().flatten())

    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mape = mean_absolute_percentage_error(all_targets, all_predictions) * 100

    return mae, mape, all_predictions, all_targets

def predict_new_model(model, feature_scaler, config, model_info, device):
    """
    Predict bandwidth for a new model configuration

    Args:
        model: Trained PyTorch model
        feature_scaler: Fitted scaler for features
        config: Dict with runtime config (workers, batch_size, ps, tensorsize)
        model_info: Dict with model architecture info (num_parameters, num_layers, etc.)
        device: PyTorch device

    Returns:
        Predicted bandwidth
    """
    # Create a dataframe with the new configuration
    new_data = {
        'tensorsize': config.get('tensorsize', 1),
        'Number of Workers': config.get('num_workers', 1),
        'Batch Size': config.get('batch_size', 32),
        'Number of PSs': config.get('num_ps', 0),
        'Data Set': config.get('dataset', 'cifar10'),
        'Pattern': config.get('pattern', 'allreduce').lower()  # Convert to lowercase
    }

    # Add model architecture information
    new_data.update({
        'Number of Parameters': model_info.get('num_parameters', 0),
        'Number of Layers': model_info.get('num_layers', 0)
        # Intentionally not including model_depth
    })

    # Convert to DataFrame
    df = pd.DataFrame([new_data])

    # Extract features
    features = extract_model_independent_features(df)

    # For debugging
    print("Features before alignment:", features.columns.tolist())

    # Get the feature names used during training
    scaler_feature_names = feature_scaler.feature_names_in_

    print("Scaler features:", scaler_feature_names.tolist())
    print("Missing from prediction:", set(scaler_feature_names) - set(features.columns))
    print("Extra in prediction:", set(features.columns) - set(scaler_feature_names))

    # Ensure all expected columns exist
    for col in scaler_feature_names:
        if col not in features.columns:
            features[col] = 0

    # Make sure we only include columns that were in the training data
    features = features[scaler_feature_names]

    # Final check
    print("Features after alignment:", features.columns.tolist())

    # Scale features
    scaled_features = feature_scaler.transform(features)

    # Convert to tensor
    input_tensor = torch.FloatTensor(scaled_features).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print("Raw model output (log scale):", output.item())

        # 안전한 역변환 (너무 큰 값 방지)
        if output.item() > 20:  # exp(20) ≈ 485 million, 충분히 큰 값
            print("Warning: Very large prediction detected, capping output")
            output = torch.tensor([20.0]).to(device)

        # 역변환
        predicted_bandwidth = torch.exp(output).item() - 1

    return predicted_bandwidth

def main():
    # Create output directory structure
    output_dir = "../results/bandwidth_predictor"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv('../data/dataset_v2.csv')
        print("Using dataset_v2.csv")
    except FileNotFoundError:
        df = pd.read_csv('../data/cnn_network_summary.csv')
        print("Using cnn_network_summary.csv")

    # Check for required columns
    required_columns = ['tensorsize', 'Number of Workers', 'Batch Size', 'Number of PSs',
                       'Model', 'Sum of Max TX+RX (MB/s)']

    for col in required_columns:
        if col not in df.columns:
            print(f"WARNING: Required column '{col}' is not in the dataset!")

    # Extract model-independent features
    features = extract_model_independent_features(df)

    # Log transform the target (bandwidth)
    targets = np.log1p(df['Sum of Max TX+RX (MB/s)'].values)

    # Scale features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)

    # Save feature scaler for later use
    scaler_path = os.path.join(run_dir, "feature_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)

    # Save feature names
    feature_names_path = os.path.join(run_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(features.columns.tolist(), f)

    # Split data into training (80%) and test (20%) sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, targets, test_size=0.2, random_state=42
    )

    # Further split training into train and validation (80/20 split of the 80% training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = CommDataset(X_train, y_train)
    val_dataset = CommDataset(X_val, y_val)
    test_dataset = CommDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train model
    input_dim = X_train.shape[1]
    model = train_model(train_loader, val_loader, input_dim, device, run_dir)

    # Evaluate on test set
    mae, mape, predictions, targets = evaluate_model(model, test_loader, device)
    print(f"Test MAE: {mae:.2f} MB/s")
    print(f"Test MAPE: {mape:.2f}%")

    # Save evaluation results
    eval_results = {
        'mae': mae,
        'mape': mape,
        'timestamp': timestamp,
        'num_features': input_dim,
        'feature_names': features.columns.tolist()
    }

    with open(os.path.join(run_dir, "evaluation_results.json"), 'w') as f:
        json.dump(eval_results, f)

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual Bandwidth (MB/s)')
    plt.ylabel('Predicted Bandwidth (MB/s)')
    plt.title('Predicted vs Actual Bandwidth')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "predictions_vs_actual.png"))

    # Example usage of the prediction function for a new model
    print("\nExample prediction for a new model:")

    new_config = {
        'tensorsize': 10000000,  # 10MB tensor
        'num_workers': 4,
        'batch_size': 64,
        'num_ps': 0,  # AllReduce
        'dataset': 'imagenet',
        'pattern': 'allreduce'  # Changed to lowercase
    }

    new_model_info = {
        'num_parameters': 25000000,  # 25M params
        'num_layers': 50
        # Note: model_depth intentionally removed
    }

    predicted_bw = predict_new_model(model, feature_scaler, new_config, new_model_info, device)
    print(f"Predicted bandwidth: {predicted_bw:.2f} MB/s")

    # Create a simple prediction script
    prediction_script = """
import torch
import pickle
import json
import pandas as pd
import numpy as np

def predict_bandwidth(config, model_info):
    \"\"\"
    Predict communication bandwidth for a given model and configuration.

    Args:
        config: Dict with runtime config (workers, batch_size, ps, tensorsize)
        model_info: Dict with model architecture info (num_parameters, num_layers, etc.)

    Returns:
        Predicted bandwidth in MB/s
    \"\"\"
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
    \"\"\"Create model-independent features that can be used for any model\"\"\"
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
    """

    # Save the prediction script
    with open(os.path.join(run_dir, "predict.py"), 'w') as f:
        f.write(prediction_script)

    print(f"All outputs saved to {run_dir}")

if __name__ == '__main__':
    main()
