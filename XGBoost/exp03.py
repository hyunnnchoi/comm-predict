import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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
    features['ring_volume'] = df['tensorsize'] * 2 * (N-1) / N * features['is_ring_allreduce']
    features['ring_steps'] = 2 * (N-1) * features['is_ring_allreduce']
    features['ring_bandwidth_per_step'] = features['ring_volume'] / features['ring_steps'].clip(lower=1)

    # PS 특성 (PS>0일 때)
    ps_mask = 1 - features['is_ring_allreduce']
    features['ps_volume'] = df['tensorsize'] * N * 2 * ps_mask
    features['ps_worker_ratio'] = (N / df['Number of PSs'].clip(lower=1)) * ps_mask

    # 모델 구조 특성
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

# 데이터 로드 및 특성 생성
df = pd.read_csv('../data/cnn_network_summary.csv')
X = create_features(df)
y = df['Sum of Max TX+RX (MB/s)']

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

results = []
unique_models = df['Model'].unique()

# Leave-One-Model-Out 평가: 각 모델을 테스트셋으로 두고 나머지로 학습
for left_out_model in unique_models:
    print(f"\n===== Leave-One-Model-Out: Testing on {left_out_model} =====")
    test_mask = df['Model'] == left_out_model
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_test  = X[test_mask]
    y_test  = y[test_mask]

    # XGBRegressor 생성 (early_stopping_rounds는 생성자에서 설정)
    model = xgb.XGBRegressor(early_stopping_rounds=30, **xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )

    # 예측 및 평가 지표 계산
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_mae  = mean_absolute_error(y_train, train_pred)
    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    test_mae   = mean_absolute_error(y_test, test_pred)
    test_mape  = mean_absolute_percentage_error(y_test, test_pred) * 100

    print(f"Train MAE: {train_mae:.2f} MB/s, Train MAPE: {train_mape:.2f}%")
    print(f"Test MAE: {test_mae:.2f} MB/s, Test MAPE: {test_mape:.2f}%")

    # 상세 테스트 결과 출력 (옵션)
    test_df = df[test_mask]
    for i, (pred, actual) in enumerate(zip(test_pred, y_test)):
        error = abs(pred - actual) / actual * 100
        print(f"Sample {i}: Batch Size: {test_df['Batch Size'].iloc[i]}, "
              f"Workers: {test_df['Number of Workers'].iloc[i]}, "
              f"PS: {test_df['Number of PSs'].iloc[i]} -> "
              f"Predicted: {pred:.2f} MB/s, Actual: {actual:.2f} MB/s, Error: {error:.2f}%")

    results.append({
        'left_out_model': left_out_model,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'test_mae': test_mae,
        'test_mape': test_mape
    })

# 평가 결과 요약 출력
results_df = pd.DataFrame(results)
print("\n===== Leave-One-Model-Out Evaluation Summary =====")
print(results_df)
