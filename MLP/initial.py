import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler


class CommDataset(Dataset):
    def __init__(self, csv_path='../data/cnn_network_summary.csv'):

        df = pd.read_csv(csv_path)

        # [2] Problem: 입력 데이터 오류 가능성, Answer: 검증 cout 추가
        print("데이터 통계:", df.describe())
        print("NaN 값 확인:", df.isna().sum())

        self.model_names = df['Model']
        self.tensorsize = df['tensorsize']
        self.num_workers = df['Number of Workers']
        self.comm_volume = df['Sum of Max TX+RX (MB/s)']

        self.label_encoder = LabelEncoder()
        self.model_encoded = self.label_encoder.fit_transform(self.model_names)

        self.features = np.column_stack((
            self.model_encoded,
            self.tensorsize, # Problem [1] 여기서 tensorsize가 매우 클 수 있음. 정규화 필요?
            self.num_workers
        ))
 # [1] Solution: tensorsize, bandwidth 클 수 있어서 avg 0, variance 1로 정규화
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()

        self.features = self.scaler_features.fit_transform(self.features)
        self.comm_volume = self.scaler_target.fit_transform(self.comm_volume.values.reshape(-1, 1))

        self.features = torch.FloatTensor(self.features)
        self.comm_volume = torch.FloatTensor(self.comm_volume)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.comm_volume[idx]

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

        # [3] Gradient clipping 추가, 파라미터 업데이터 큰 값 튈 수 있음 -> 업데이트 크기 upper bound 설정
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = CommDataset()
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = CommunicationPredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # [4] lr 0.01 -> 0.0001

    num_epochs = 500
    for epoch in range(num_epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

if __name__ == '__main__':
    main()
