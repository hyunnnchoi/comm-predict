import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 모델 아키텍처 특성 정의
model_architecture_features = {
    'alexnet': {'depth': 8, 'sequential_ratio': 0.8, 'branch_factor': 1, 'param_density': 0.6},
    'resnet110': {'depth': 110, 'sequential_ratio': 0.2, 'branch_factor': 2, 'param_density': 0.4},
    'resnet44': {'depth': 44, 'sequential_ratio': 0.2, 'branch_factor': 2, 'param_density': 0.4},
    'resnet56': {'depth': 56, 'sequential_ratio': 0.2, 'branch_factor': 2, 'param_density': 0.4},
    'densenet40_k12': {'depth': 40, 'sequential_ratio': 0.1, 'branch_factor': 40, 'param_density': 0.3},
    'densenet100_k12': {'depth': 100, 'sequential_ratio': 0.1, 'branch_factor': 100, 'param_density': 0.3},
    'googlenet': {'depth': 22, 'sequential_ratio': 0.3, 'branch_factor': 3, 'param_density': 0.5},
    'vgg16': {'depth': 16, 'sequential_ratio': 0.9, 'branch_factor': 1, 'param_density': 0.7},
    'inception3': {'depth': 48, 'sequential_ratio': 0.2, 'branch_factor': 4, 'param_density': 0.4},
    'resnet50': {'depth': 50, 'sequential_ratio': 0.2, 'branch_factor': 2, 'param_density': 0.4},
    'bert': {'depth': 12, 'sequential_ratio': 0.5, 'branch_factor': 3, 'param_density': 0.8},
    'gpt2': {'depth': 12, 'sequential_ratio': 0.6, 'branch_factor': 2, 'param_density': 0.9}
}

# XGBoost 파라미터
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': ['mae', 'mape'],
    'max_depth': 4,
    'learning_rate': 0.03,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.5,
    'random_state': 42
}

def create_features(df):
    features = pd.DataFrame()

    # 기본 특성
    features['tensor_size'] = df['tensorsize']
    features['num_workers'] = df['Number of Workers']
    features['batch_size'] = df['Batch Size']
    features['num_ps'] = df['Number of PSs']

    # 통신 패턴 구분 및 특성
    features['is_ring_allreduce'] = (df['Number of PSs'] == 0).astype(int)
    N = df['Number of Workers']

    # Ring AllReduce 특성
    features['ring_volume'] = df['tensorsize'] * 2 * (N-1) / N * features['is_ring_allreduce']
    features['ring_steps'] = 2 * (N-1) * features['is_ring_allreduce']
    features['ring_bandwidth_per_step'] = features['ring_volume'] / features['ring_steps'].clip(lower=1)

    # PS 특성
    ps_mask = 1 - features['is_ring_allreduce']
    features['ps_volume'] = df['tensorsize'] * N * 2 * ps_mask
    features['ps_worker_ratio'] = (N / df['Number of PSs'].clip(lower=1)) * ps_mask

    # 모델 구조 특성
    def get_model_feature(model_name, feature):
        return model_architecture_features.get(model_name, {}).get(feature, 0)

    features['model_depth'] = df['Model'].apply(lambda x: get_model_feature(x, 'depth'))
    features['sequential_ratio'] = df['Model'].apply(lambda x: get_model_feature(x, 'sequential_ratio'))
    features['branch_factor'] = df['Model'].apply(lambda x: get_model_feature(x, 'branch_factor'))
    features['param_density'] = df['Model'].apply(lambda x: get_model_feature(x, 'param_density'))

    # 새롭게 추가된 Feature 반영
    features['num_parameters'] = df['Number of Parameters'].astype(float)
    features['num_layers'] = df['Number of Layers'].astype(float)

    # 통신/계산 특성
    features['compute_intensity'] = features['param_density'] * features['tensor_size']
    features['comm_intensity'] = features['tensor_size'] * N / features['batch_size']

    # Feature interactions
    features['ring_worker_interaction'] = features['ring_volume'] * np.log2(N)
    features['ps_worker_interaction'] = features['ps_volume'] * features['ps_worker_ratio']

    # 로그 변환 특성
    features['log_tensor_size'] = np.log1p(df['tensorsize'])
    features['log_batch_size'] = np.log(df['Batch Size'])
    features['log_workers'] = np.log2(N)

    # 데이터셋 크기
    features['dataset_size'] = df['Data Set'].map({'cifar10': 1, 'imagenet': 10, 'squad': 5})

    # Pattern을 범주형 변수로 변환 (원-핫 인코딩)
    pattern_dummies = pd.get_dummies(df['Pattern'], prefix='pattern')
    features = pd.concat([features, pattern_dummies], axis=1)

    return features

# 데이터 로드 및 전처리
# CSV 파일 직접 지정 (데이터 파일 경로를 직접 지정하세요)

# CSV 파일을 명시적으로 로드하고 칼럼 이름을 보존함
df = pd.read_csv('../data/dataset_v2.csv')

# 컬럼 이름에 공백이 있는지 확인
print("Original column names:", df.columns.tolist())

# 컬럼 이름에 있을 수 있는 공백 제거
df.columns = df.columns.str.strip()
print("Cleaned column names:", df.columns.tolist())

# 필요한 컬럼이 있는지 확인
required_columns = ['tensorsize', 'Number of Workers', 'Batch Size', 'Number of PSs',
                   'Model', 'Number of Parameters', 'Number of Layers', 'Data Set', 'Pattern']

for col in required_columns:
    if col not in df.columns:
        print(f"WARNING: Required column '{col}' is not in the dataset!")

# 특성 생성 및 모델 학습
X = create_features(df)
y = df['Sum of Max TX+RX (MB/s)']

# Leave-One-Model-Out 평가
results = []
unique_models = df['Model'].unique()

for left_out_model in unique_models:
    test_mask = df['Model'] == left_out_model
    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = xgb.XGBRegressor(early_stopping_rounds=30, **xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

    results.append({'left_out_model': left_out_model, 'test_mae': test_mae, 'test_mape': test_mape})

# 결과 요약
results_df = pd.DataFrame(results)
print("\n===== Leave-One-Model-Out Evaluation Summary =====")
print(results_df)


'''
이 코드에서 사용되는 모든 특성(Feature)을 분석해보겠습니다. create_features 함수에서 생성하는 특성들과 그 계산 방식을 순서대로 정리하겠습니다:
기본 특성 (Raw Features)

tensor_size: 텐서 크기 (원본 데이터의 'tensorsize' 컬럼)
num_workers: 워커 수 (원본 데이터의 'Number of Workers' 컬럼)
batch_size: 배치 크기 (원본 데이터의 'Batch Size' 컬럼)
num_ps: 파라미터 서버 수 (원본 데이터의 'Number of PSs' 컬럼)

통신 패턴 관련 특성

is_ring_allreduce: Ring AllReduce 방식 사용 여부 (파라미터 서버가 0개이면 1, 아니면 0)
ring_volume: Ring AllReduce 방식에서의 통신량 (= tensor_size * 2 * (N-1) / N * is_ring_allreduce)
ring_steps: Ring AllReduce 방식에서의 통신 단계 수 (= 2 * (N-1) * is_ring_allreduce)
ring_bandwidth_per_step: 단계당 대역폭 (= ring_volume / ring_steps)
ps_volume: 파라미터 서버 방식에서의 통신량 (= tensor_size * N * 2 * ps_mask)
ps_worker_ratio: 워커 대 PS 비율 (= N / num_ps * ps_mask)

모델 구조 관련 특성 (사전 정의된 model_architecture_features 사용)

model_depth: 모델의 깊이
sequential_ratio: 순차적 연산 비율
branch_factor: 분기 요소
param_density: 파라미터 밀도

추가 모델 관련 특성

num_parameters: 파라미터 수 (원본 데이터의 'Number of Parameters' 컬럼)
num_layers: 레이어 수 (원본 데이터의 'Number of Layers' 컬럼)

계산 및 통신 강도 특성

compute_intensity: 계산 강도 (= param_density * tensor_size)
comm_intensity: 통신 강도 (= tensor_size * N / batch_size)

특성 상호작용

ring_worker_interaction: Ring 방식에서의 워커 상호작용 (= ring_volume * log2(N))
ps_worker_interaction: PS 방식에서의 워커 상호작용 (= ps_volume * ps_worker_ratio)

로그 변환 특성

log_tensor_size: 텐서 크기의 로그 변환 (= log1p(tensorsize))
log_batch_size: 배치 크기의 로그 변환 (= log(batch_size))
log_workers: 워커 수의 로그 변환 (= log2(N))

데이터셋 관련 특성

dataset_size: 데이터셋 크기 (cifar10=1, imagenet=10, squad=5로 매핑)

범주형 특성 (원-핫 인코딩)

pattern_*: 통신 패턴을 나타내는 원-핫 인코딩 변수들 (원본 데이터의 'Pattern' 컬럼에서 생성)

이 특성들은 XGBoost 모델의 입력으로 사용되며, 분산 학습 시스템의 통신 성능(Sum of Max TX+RX (MB/s))을 예측하는 데 활용됩니다. 특성들은 통신 패턴, 모델 구조, 시스템 구성 등 다양한 측면을 고려하여 설계되었습니다.


'''
