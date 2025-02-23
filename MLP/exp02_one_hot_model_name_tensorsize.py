# 개선 사항
# 모델명 LabelEncoder 대신 one-hot encoding 사용
# Dropout 추가 - Overfit 방지
# 모델 구조 임베딩 기반으로 변경
# tensorsize와 통신량 상관관계 잘 학습하도록 구조 설계

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. 데이터 로드 및 분석
df = pd.read_csv('../data/cnn_network_summary.csv')

# 텐서사이즈와 통신량 관계 시각화
plt.figure(figsize=(12, 6))
for model in df['Model'].unique():
    data = df[df['Model'] == model]
    plt.scatter(data['tensorsize'],
                data['Sum of Max TX+RX (MB/s)'],
                label=model,
                alpha=0.6)
plt.xlabel('Tensor Size')
plt.ylabel('Communication Volume (MB/s)')
plt.title('Tensor Size vs Communication Volume by Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('tensorsize_vs_comm.png')
plt.close()

# 상관관계 분석
correlation = df['tensorsize'].corr(df['Sum of Max TX+RX (MB/s)'])
print(f"\nCorrelation between tensorsize and communication volume: {correlation:.3f}")

# 모델별 통계
print("\nModel Statistics:")
for model in df['Model'].unique():
    data = df[df['Model'] == model]
    print(f"\n{model}:")
    print(f"Mean bandwidth: {data['Sum of Max TX+RX (MB/s)'].mean():.2f}")
    print(f"Std bandwidth: {data['Sum of Max TX+RX (MB/s)'].std():.2f}")
    print(f"Tensor size: {data['tensorsize'].iloc[0]}")

# 2. One-hot encoding을 사용한 새로운 모델
class CommDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CommunicationPredictor(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.model_embedding = nn.Linear(num_models, 16)  # one-hot -> embedding

        self.main_network = nn.Sequential(
            nn.Linear(18, 64),  # 16 (embedding) + 2 (tensorsize, workers)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x의 구조: [batch_size, num_features]
        # 처음 num_models개는 one-hot encoding
        model_features = x[:, :-2]  # one-hot part
        other_features = x[:, -2:]  # tensorsize and workers

        model_embedded = self.model_embedding(model_features)
        combined = torch.cat([model_embedded, other_features], dim=1)

        return self.main_network(combined)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device, scaler_target):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            pred_rescaled = scaler_target.inverse_transform(outputs.cpu().numpy())
            actual_rescaled = scaler_target.inverse_transform(labels.cpu().numpy())

            predictions.extend(pred_rescaled)
            actuals.extend(actual_rescaled)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    for pred, actual in zip(predictions, actuals):
        error_percent = ((pred - actual) / actual) * 100
        print(f'Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}, Error: {error_percent[0]:.2f}%')

    return predictions, actuals

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 준비
    df = pd.read_csv('../data/cnn_network_summary.csv')

    # One-hot encoding for models
    model_dummies = pd.get_dummies(df['Model'])

    # resnet110을 테스트 세트로 분리
    test_mask = df['Model'] == 'resnet110'
    train_df = df[~test_mask]
    test_df = df[test_mask]

    # 특성 준비
    train_model_dummies = model_dummies[~test_mask]
    test_model_dummies = model_dummies[test_mask]

    # 특성 행렬 생성
    train_features = np.concatenate([
        train_model_dummies.values,
        train_df[['tensorsize', 'Number of Workers']].values
    ], axis=1)

    test_features = np.concatenate([
        test_model_dummies.values,
        test_df[['tensorsize', 'Number of Workers']].values
    ], axis=1)

    # 스케일링
    scaler_target = StandardScaler()
    train_targets = scaler_target.fit_transform(train_df['Sum of Max TX+RX (MB/s)'].values.reshape(-1, 1))
    test_targets = scaler_target.transform(test_df['Sum of Max TX+RX (MB/s)'].values.reshape(-1, 1))

    # 데이터로더 생성
    train_dataset = CommDataset(train_features, train_targets)
    test_dataset = CommDataset(test_features, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # 모델 학습
    num_models = len(model_dummies.columns)
    model = CommunicationPredictor(num_models).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 500
    for epoch in range(num_epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    print("\nTesting on resnet110 data:")
    predictions, actuals = evaluate(model, test_loader, criterion, device, scaler_target)

if __name__ == '__main__':
    main()
