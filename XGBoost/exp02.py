import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost.callback import EarlyStopping

# 모델 아키텍처 특성 정의
model_architecture_features = {
    'alexnet': {
        'depth': 8,
        'sequential_ratio': 0.8,
        'branch_factor': 1,
        'param_density': 0.6
    },
    'resnet110': {
        'depth': 110,
        'sequential_ratio': 0.2,
        'branch_factor': 2,
        'param_density': 0.4
    },
    'resnet44': {
        'depth': 44,
        'sequential_ratio': 0.2,
        'branch_factor': 2,
        'param_density': 0.4
    },
    'resnet56': {
        'depth': 56,
        'sequential_ratio': 0.2,
        'branch_factor': 2,
        'param_density': 0.4
    },
    'densenet40_k12': {
        'depth': 40,
        'sequential_ratio': 0.1,
        'branch_factor': 40,
        'param_density': 0.3
    },
    'densenet100_k12': {
        'depth': 100,
        'sequential_ratio': 0.1,
        'branch_factor': 100,
        'param_density': 0.3
    },
    'googlenet': {
        'depth': 22,
        'sequential_ratio': 0.3,
        'branch_factor': 3,
        'param_density': 0.5
    },
    'vgg16': {
        'depth': 16,
        'sequential_ratio': 0.9,
        'branch_factor': 1,
        'param_density': 0.7
    },
    'inception3': {
        'depth': 48,
        'sequential_ratio': 0.2,
        'branch_factor': 4,
        'param_density': 0.4
    },
    'bert': {
        'depth': 12,
        'sequential_ratio': 0.5,
        'branch_factor': 3,
        'param_density': 0.8
    },
    'gpt2': {
        'depth': 12,
        'sequential_ratio': 0.6,
        'branch_factor': 2,
        'param_density': 0.9
    }
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

    # Ring AllReduce 특성 (PS=0일 때)
    features['ring_volume'] = df['tensorsize'] * 2 * (N-1)/N * features['is_ring_allreduce']
    features['ring_steps'] = 2 * (N-1) * features['is_ring_allreduce']
    features['ring_bandwidth_per_step'] = features['ring_volume'] / features['ring_steps'].clip(lower=1)

    # PS 특성 (PS>0일 때)
    ps_mask = 1 - features['is_ring_allreduce']
    features['ps_volume'] = df['tensorsize'] * N * 2 * ps_mask
    features['ps_worker_ratio'] = (N / df['Number of PSs'].clip(lower=1)) * ps_mask

    # 모델 구조 특성 - 더 안전한 방식으로 변경
    def get_model_feature(model_name, feature):
        if model_name in model_architecture_features:
            return model_architecture_features[model_name][feature]
        return 0  # 기본값

    features['model_depth'] = df['Model'].apply(lambda x: get_model_feature(x, 'depth'))
    features['sequential_ratio'] = df['Model'].apply(lambda x: get_model_feature(x, 'sequential_ratio'))
    features['branch_factor'] = df['Model'].apply(lambda x: get_model_feature(x, 'branch_factor'))
    features['param_density'] = df['Model'].apply(lambda x: get_model_feature(x, 'param_density'))

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
    features['dataset_size'] = df['Data Set'].map({
        'cifar10': 1,
        'imagenet': 10,
        'squad': 5
    })

    return features
# 데이터 로드
df = pd.read_csv('../data/cnn_network_summary.csv')

# 특성 생성
X = create_features(df)
y = df['Sum of Max TX+RX (MB/s)']

# 테스트 셋 분리 (resnet110)
test_mask = df['Model'] == 'resnet110'
X_train = X[~test_mask]
y_train = y[~test_mask]
X_test = X[test_mask]
y_test = y[test_mask]

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

# 모델 학습
model = xgb.XGBRegressor(**xgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=30,
    verbose=True
)



# 예측 및 평가
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 평가 지표 계산
train_mae = mean_absolute_error(y_train, train_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
test_mae = mean_absolute_error(y_test, test_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

print("\nTraining Results:")
print(f"MAE: {train_mae:.2f} MB/s")
print(f"MAPE: {train_mape:.2f}%")

print("\nTest Results (resnet110):")
print(f"MAE: {test_mae:.2f} MB/s")
print(f"MAPE: {test_mape:.2f}%")

# 상세 예측 결과
print("\nDetailed Test Predictions:")
test_data = df[test_mask]
for i, (pred, actual) in enumerate(zip(test_pred, y_test)):
    error = abs(pred - actual) / actual * 100
    print(f"Batch Size: {test_data['Batch Size'].iloc[i]}, "
          f"Workers: {test_data['Number of Workers'].iloc[i]}, "
          f"PS: {test_data['Number of PSs'].iloc[i]}")
    print(f"Predicted: {pred:.2f} MB/s, Actual: {actual:.2f} MB/s, Error: {error:.2f}%")

# 특성 중요도 분석
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# 모델 저장
model.save_model('comm_predictor_v3.json')

# 통신 패턴별 특성 개선:

# Ring AllReduce와 PS 각각의 통신 특성을 정확하게 모델링
# Feature interaction 추가


# 하이퍼파라미터 조정:

# 더 보수적인 학습 (낮은 learning rate, 더 얕은 트리)
# Early stopping 추가


# 모델 특성 단순화:

# 불필요한 피처 제거
# 해석 가능한 피처 위주로 구성
