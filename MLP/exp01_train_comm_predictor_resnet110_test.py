import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class CommDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CommunicationPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layers(x)

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

            # 원래 스케일로 되돌리기
            pred_rescaled = scaler_target.inverse_transform(outputs.cpu().numpy())
            actual_rescaled = scaler_target.inverse_transform(labels.cpu().numpy())

            predictions.extend(pred_rescaled)
            actuals.extend(actual_rescaled)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 예측 결과 출력
    for pred, actual in zip(predictions, actuals):
        error_percent = ((pred - actual) / actual) * 100
        print(f'Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}, Error: {error_percent[0]:.2f}%')

    return predictions, actuals

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로드
    df = pd.read_csv('../data/cnn_network_summary.csv')

    print("데이터 통계:", df.describe())
    print("NaN 값 확인:", df.isna().sum())

    # LabelEncoder를 전체 데이터에 먼저 적용
    label_encoder = LabelEncoder()
    all_models_encoded = label_encoder.fit_transform(df['Model'])

    # resnet110을 테스트 세트로 분리
    test_mask = df['Model'] == 'resnet110'
    train_df = df[~test_mask]
    test_df = df[test_mask]

    train_models_encoded = all_models_encoded[~test_mask]
    test_models_encoded = all_models_encoded[test_mask]

    # 특성 행렬 생성
    train_features = np.column_stack((
        train_models_encoded,
        train_df['tensorsize'],
        train_df['Number of Workers']
    ))

    test_features = np.column_stack((
        test_models_encoded,
        test_df['tensorsize'],
        test_df['Number of Workers']
    ))

    # 스케일링
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    train_features_scaled = scaler_features.fit_transform(train_features)
    test_features_scaled = scaler_features.transform(test_features)

    train_targets = scaler_target.fit_transform(train_df['Sum of Max TX+RX (MB/s)'].values.reshape(-1, 1))
    test_targets = scaler_target.transform(test_df['Sum of Max TX+RX (MB/s)'].values.reshape(-1, 1))

    # 데이터셋 및 데이터로더 생성
    train_dataset = CommDataset(train_features_scaled, train_targets)
    test_dataset = CommDataset(test_features_scaled, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # 모델 학습
    model = CommunicationPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 500
    for epoch in range(num_epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    # 테스트 세트에서 평가
    print("\nTesting on resnet110 data:")
    predictions, actuals = evaluate(model, test_loader, criterion, device, scaler_target)

if __name__ == '__main__':
    main()
