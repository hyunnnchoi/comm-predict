import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import time
import os

class CommDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CommunicationPredictor(nn.Module):
    def __init__(self, num_models, num_datasets):
        super().__init__()

        # Separate embeddings for models and datasets
        self.model_embedding = nn.Linear(num_models, 32)
        self.dataset_embedding = nn.Linear(num_datasets, 16)

        # Feature processing networks
        self.tensor_network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        self.worker_network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        self.batch_network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        self.ps_network = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )

        # Main network
        combined_size = 32 + 16 + 16 + 16 + 16 + 8  # All embeddings and processed features

        self.main_network = nn.Sequential(
            nn.Linear(combined_size, 128),
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
        # Split input features
        model_onehot = x[:, :self.model_embedding.in_features]
        dataset_onehot = x[:, self.model_embedding.in_features:self.model_embedding.in_features + self.dataset_embedding.in_features]
        tensor_size = x[:, -4:-3]
        num_workers = x[:, -3:-2]
        batch_size = x[:, -2:-1]
        num_ps = x[:, -1:]

        # Process each feature type
        model_embedded = self.model_embedding(model_onehot)
        dataset_embedded = self.dataset_embedding(dataset_onehot)
        tensor_processed = self.tensor_network(tensor_size)
        worker_processed = self.worker_network(num_workers)
        batch_processed = self.batch_network(batch_size)
        ps_processed = self.ps_network(num_ps)

        # Combine all features
        combined = torch.cat([
            model_embedded,
            dataset_embedded,
            tensor_processed,
            worker_processed,
            batch_processed,
            ps_processed
        ], dim=1)

        return self.main_network(combined)

def prepare_data(df, model_columns=None, dataset_columns=None):
    """
    Prepare features with consistent dimensions across train/test splits

    Args:
        df: DataFrame to process
        model_columns: Optional list of all model columns for consistent one-hot encoding
        dataset_columns: Optional list of all dataset columns for consistent one-hot encoding
    """
    # Convert batch size to log scale using .loc to avoid SettingWithCopyWarning
    df.loc[:, 'log_batch_size'] = np.log(df['Batch Size'])

    # One-hot encoding for models and datasets with consistent dimensions
    if model_columns is not None:
        # Create dummy variables with predetermined columns
        model_dummies = pd.get_dummies(df['Model'])
        # Add missing columns with zeros
        for col in model_columns:
            if col not in model_dummies.columns:
                model_dummies[col] = 0
        # Ensure columns are in the right order
        model_dummies = model_dummies.reindex(columns=model_columns, fill_value=0)
    else:
        model_dummies = pd.get_dummies(df['Model'])
        model_columns = model_dummies.columns.tolist()

    if dataset_columns is not None:
        # Create dummy variables with predetermined columns
        dataset_dummies = pd.get_dummies(df['Data Set'])
        # Add missing columns with zeros
        for col in dataset_columns:
            if col not in dataset_dummies.columns:
                dataset_dummies[col] = 0
        # Ensure columns are in the right order
        dataset_dummies = dataset_dummies.reindex(columns=dataset_columns, fill_value=0)
    else:
        dataset_dummies = pd.get_dummies(df['Data Set'])
        dataset_columns = dataset_dummies.columns.tolist()

    # Log transform tensorsize and bandwidth
    log_tensorsize = np.log1p(df['tensorsize'])
    log_bandwidth = np.log1p(df['Sum of Max TX+RX (MB/s)'])

    # Ensure all arrays are float32 type for PyTorch compatibility
    model_dummies_array = model_dummies.values.astype(np.float32)
    dataset_dummies_array = dataset_dummies.values.astype(np.float32)
    log_tensorsize_array = log_tensorsize.values.reshape(-1, 1).astype(np.float32)
    workers_array = df['Number of Workers'].values.reshape(-1, 1).astype(np.float32)
    batch_size_array = df['log_batch_size'].values.reshape(-1, 1).astype(np.float32)
    ps_array = df['Number of PSs'].values.reshape(-1, 1).astype(np.float32)

    # Combine features
    features = np.concatenate([
        model_dummies_array,
        dataset_dummies_array,
        log_tensorsize_array,
        workers_array,
        batch_size_array,
        ps_array
    ], axis=1)

    return features, log_bandwidth.values.astype(np.float32), model_columns, dataset_columns

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Ensure targets shape matches outputs for loss calculation
        targets = targets.view(-1, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Ensure targets shape matches outputs for loss calculation
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Convert back from log scale
            pred_exp = torch.exp(outputs) - 1
            actual_exp = torch.exp(targets) - 1

            predictions.extend(pred_exp.cpu().numpy().flatten())
            actuals.extend(actual_exp.cpu().numpy().flatten())

    val_loss = total_loss / len(val_loader)

    return val_loss, np.array(predictions), np.array(actuals)

def main():
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)

    # Generate timestamp for file naming
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = "MLP"
    results_filename = f"../results/{model_name}_LOOCV_{timestamp}.csv"

    # Load data
    df = pd.read_csv('../data/cnn_network_summary.csv')

    # Track overall results
    all_results = []

    # Get unique models for LOOCV
    unique_models = df['Model'].unique()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Always prepare the FULL dataset once to get consistent feature dimensions
    full_features, _, model_cols, dataset_cols = prepare_data(df)
    num_models = len(model_cols)
    num_datasets = len(dataset_cols)

    # This is important for one-hot encoded columns to have consistent dimensions
    # across all train/test splits

    # Leave-One-Model-Out evaluation
    for left_out_model in unique_models:
        print(f"\n===== Leave-One-Model-Out: Testing on {left_out_model} =====")

        # Split data
        test_mask = df['Model'] == left_out_model
        train_df = df[~test_mask]
        test_df = df[test_mask]

        # Prepare features and targets with consistent dimensions
        # Pass the full model_cols and dataset_cols to maintain consistent dimensions
        train_features, train_targets, _, _ = prepare_data(train_df, model_cols, dataset_cols)
        test_features, test_targets, _, _ = prepare_data(test_df, model_cols, dataset_cols)

        # Create datasets
        train_dataset = CommDataset(train_features, train_targets)
        test_dataset = CommDataset(test_features, test_targets)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # Initialize model for this fold
        model = CommunicationPredictor(num_models, num_datasets).to(device)

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

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)

            # Use training loss for scheduler since we don't have a separate validation set in LOOCV
            scheduler.step(train_loss)

            if train_loss < best_val_loss:
                best_val_loss = train_loss
                patience_counter = 0
                # Save best model for this fold
                torch.save(model.state_dict(), f'best_model_{left_out_model}.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        # Load best model for this fold
        model.load_state_dict(torch.load(f'best_model_{left_out_model}.pth'))

        # Evaluate on training data
        _, train_pred, train_actual = validate(model, train_loader, criterion, device)
        train_mae = mean_absolute_error(train_actual, train_pred)
        train_mape = mean_absolute_percentage_error(train_actual, train_pred) * 100

        # Evaluate on test data (left-out model)
        test_loss, test_pred, test_actual = validate(model, test_loader, criterion, device)
        test_mae = mean_absolute_error(test_actual, test_pred)
        test_mape = mean_absolute_percentage_error(test_actual, test_pred) * 100

        print(f"Train MAE: {train_mae:.2f} MB/s, Train MAPE: {train_mape:.2f}%")
        print(f"Test MAE: {test_mae:.2f} MB/s, Test MAPE: {test_mape:.2f}%")

        # Detailed predictions
        test_details = []
        for i, (pred, actual) in enumerate(zip(test_pred, test_actual)):
            error = ((pred - actual) / actual) * 100
            print(f'Sample {i}: Batch Size: {test_df["Batch Size"].iloc[i]}, '
                  f'Workers: {test_df["Number of Workers"].iloc[i]}, '
                  f'PS: {test_df["Number of PSs"].iloc[i]} -> '
                  f'Predicted: {pred:.2f} MB/s, Actual: {actual:.2f} MB/s, Error: {abs(error):.2f}%')

                # Store detailed prediction with all relevant info
            test_details.append({
                'model': left_out_model,
                'dataset': test_df['Data Set'].iloc[i],
                'tensorsize': test_df['tensorsize'].iloc[i],
                'batch_size': test_df['Batch Size'].iloc[i],
                'workers': test_df['Number of Workers'].iloc[i],
                'ps': test_df['Number of PSs'].iloc[i],
                'predicted': pred,
                'actual': actual,
                'error_pct': abs(error)
            })

        # Add results for this fold
        all_results.append({
            'left_out_model': left_out_model,
            'train_mae': train_mae,
            'train_mape': train_mape,
            'test_mae': test_mae,
            'test_mape': test_mape
        })

        # Save detailed predictions to CSV
        details_df = pd.DataFrame(test_details)
        details_filename = f"../results/{model_name}_{left_out_model}_details_{timestamp}.csv"
        details_df.to_csv(details_filename, index=False)
        print(f"Detailed predictions saved to {details_filename}")

    # Create DataFrame from all results
    results_df = pd.DataFrame(all_results)

    # Add average performance row
    avg_row = {
        'left_out_model': 'AVERAGE',
        'train_mae': results_df['train_mae'].mean(),
        'train_mape': results_df['train_mape'].mean(),
        'test_mae': results_df['test_mae'].mean(),
        'test_mape': results_df['test_mape'].mean()
    }
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save results to CSV
    results_df.to_csv(results_filename, index=False)
    print(f"\n===== Leave-One-Model-Out Evaluation Summary =====")
    print(results_df)
    print(f"\nResults saved to {results_filename}")

if __name__ == '__main__':
    main()
