import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

def prepare_data(df):
    # Convert batch size to log scale
    df['log_batch_size'] = np.log(df['Batch Size'])

    # One-hot encoding for models and datasets
    model_dummies = pd.get_dummies(df['Model'])
    dataset_dummies = pd.get_dummies(df['Data Set'])

    # Log transform tensorsize and bandwidth
    log_tensorsize = np.log1p(df['tensorsize'])
    log_bandwidth = np.log1p(df['Sum of Max TX+RX (MB/s)'])

    # Combine features
    features = np.concatenate([
        model_dummies.values,
        dataset_dummies.values,
        log_tensorsize.values.reshape(-1, 1),
        df['Number of Workers'].values.reshape(-1, 1),
        df['log_batch_size'].values.reshape(-1, 1),
        df['Number of PSs'].values.reshape(-1, 1)
    ], axis=1)

    return features, log_bandwidth.values

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
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
    # Load data
    df = pd.read_csv('../data/cnn_network_summary.csv')

    # Prepare data
    features, targets = prepare_data(df)

    # Split data
    test_mask = df['Model'] == 'resnet110'
    val_mask = df['Model'].isin(['resnet44', 'resnet56'])
    train_mask = ~(val_mask | test_mask)

    # Create datasets
    train_dataset = CommDataset(features[train_mask], targets[train_mask])
    val_dataset = CommDataset(features[val_mask], targets[val_mask])
    test_dataset = CommDataset(features[test_mask], targets[test_mask])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    num_models = len(pd.get_dummies(df['Model']).columns)
    num_datasets = len(pd.get_dummies(df['Data Set']).columns)
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
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pred, val_actual = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            val_mape = np.mean(np.abs((val_pred - val_actual) / val_actual)) * 100
            print(f'Validation MAPE: {val_mape:.2f}%')

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))

    # Final evaluation
    print("\nTesting on resnet110 data:")
    test_loss, test_pred, test_actual = validate(model, test_loader, criterion, device)

    test_mape = np.mean(np.abs((test_pred - test_actual) / test_actual)) * 100
    test_mae = np.mean(np.abs(test_pred - test_actual))

    print(f"\nTest Results:")
    print(f"MAE: {test_mae:.2f} MB/s")
    print(f"MAPE: {test_mape:.2f}%")

    # Detailed predictions
    for pred, actual in zip(test_pred, test_actual):
        error = ((pred - actual) / actual) * 100
        print(f'Predicted: {pred:.2f} MB/s, Actual: {actual:.2f} MB/s, Error: {error:.2f}%')

if __name__ == '__main__':
    main()

# 주요 개선 사항을 설명드리면:

# 특성 엔지니어링


# Dataset을 별도의 특성으로 추가 (cifar10, imagenet, squad)
# Batch size를 log scale로 변환
# Parameter Server 수를 별도 특성으로 활용


# 모델 구조 개선


# 각 입력 타입별로 별도의 처리 네트워크 구성:

# Model embedding
# Dataset embedding
# Tensor size network
# Worker count network
# Batch size network
# PS count network


# 각 네트워크의 출력을 결합하여 최종 예측


# 학습 전략


# Batch size 증가 (8 → 16)
# Epoch 수 감소 (500 → 300)
# 더 공격적인 learning rate 감소 (patience 20 → 10)
# Weight decay 유지


# 데이터 분할


# Test set: resnet110
# Validation set: resnet44, resnet56
# Train set: 나머지 모델들

# 이 버전은 각 특성의 고유한 패턴을 더 잘 포착할 수 있도록 설계되었습니다. 특히:

# Dataset별 특성을 반영
# Worker 수와 PS 수의 관계를 명시적으로 모델링
# Batch size의 비선형성을 log 변환으로 처리
