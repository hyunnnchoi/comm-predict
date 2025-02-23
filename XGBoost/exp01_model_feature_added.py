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
        'conv_ratio': 0.7,
        'fc_ratio': 0.3,
        'param_density': 0.6
    },
    'resnet110': {
        'depth': 110,
        'sequential_ratio': 0.2,
        'branch_factor': 2,
        'conv_ratio': 0.95,
        'fc_ratio': 0.05,
        'param_density': 0.4
    },
    'resnet44': {
        'depth': 44,
        'sequential_ratio': 0.2,
        'branch_factor': 2,
        'conv_ratio': 0.95,
        'fc_ratio': 0.05,
        'param_density': 0.4
    },
    'resnet56': {
        'depth': 56,
        'sequential_ratio': 0.2,
        'branch_factor': 2,
        'conv_ratio': 0.95,
        'fc_ratio': 0.05,
        'param_density': 0.4
    },
    'densenet40_k12': {
        'depth': 40,
        'sequential_ratio': 0.1,
        'branch_factor': 40,
        'conv_ratio': 0.9,
        'fc_ratio': 0.1,
        'param_density': 0.3
    },
    'densenet100_k12': {
        'depth': 100,
        'sequential_ratio': 0.1,
        'branch_factor': 100,
        'conv_ratio': 0.9,
        'fc_ratio': 0.1,
        'param_density': 0.3
    },
    'googlenet': {
        'depth': 22,
        'sequential_ratio': 0.3,
        'branch_factor': 3,
        'conv_ratio': 0.9,
        'fc_ratio': 0.1,
        'param_density': 0.5
    },
    'vgg16': {
        'depth': 16,
        'sequential_ratio': 0.9,
        'branch_factor': 1,
        'conv_ratio': 0.8,
        'fc_ratio': 0.2,
        'param_density': 0.7
    },
    'inception3': {
        'depth': 48,
        'sequential_ratio': 0.2,
        'branch_factor': 4,
        'conv_ratio': 0.95,
        'fc_ratio': 0.05,
        'param_density': 0.4
    },
    'bert': {
        'depth': 12,
        'sequential_ratio': 0.5,
        'branch_factor': 3,
        'conv_ratio': 0.0,
        'fc_ratio': 1.0,
        'param_density': 0.8
    },
    'gpt2': {
        'depth': 12,
        'sequential_ratio': 0.6,
        'branch_factor': 2,
        'conv_ratio': 0.0,
        'fc_ratio': 1.0,
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

    # 모델 구조 특성
    features['model_depth'] = df['Model'].map({k: v['depth'] for k,v in model_architecture_features.items()})
    features['sequential_ratio'] = df['Model'].map({k: v['sequential_ratio'] for k,v in model_architecture_features.items()})
    features['branch_factor'] = df['Model'].map({k: v['branch_factor'] for k,v in model_architecture_features.items()})
    features['conv_ratio'] = df['Model'].map({k: v['conv_ratio'] for k,v in model_architecture_features.items()})
    features['fc_ratio'] = df['Model'].map({k: v['fc_ratio'] for k,v in model_architecture_features.items()})
    features['param_density'] = df['Model'].map({k: v['param_density'] for k,v in model_architecture_features.items()})

    # 도메인 지식 기반 특성
    features['tensor_per_worker'] = df['tensorsize'] / df['Number of Workers']
    features['comm_intensity'] = df['Batch Size'] * df['tensorsize'] / df['Number of Workers']
    features['compute_intensity'] = features['param_density'] * features['tensor_size']
    features['structure_complexity'] = features['branch_factor'] * features['model_depth']

    # 통신 패턴 특성
    features['all_to_all_comm'] = df['tensorsize'] * (df['Number of Workers'] - 1)
    features['ring_comm'] = df['tensorsize'] * 2
    features['log2_workers'] = np.log2(df['Number of Workers'])
    features['sqrt_workers'] = np.sqrt(df['Number of Workers'])
    features['comm_comp_ratio'] = (df['tensorsize'] * df['Number of Workers']) / df['Batch Size']

    # 데이터셋 크기 (상대적)
    features['dataset_size'] = df['Data Set'].map({
        'cifar10': 1,
        'imagenet': 10,
        'squad': 5
    })

    # 로그 변환 특성
    features['log_tensor_size'] = np.log1p(df['tensorsize'])
    features['log_batch_size'] = np.log(df['Batch Size'])
    features['log_comm_intensity'] = np.log1p(features['comm_intensity'])

    # 모델 타입 one-hot은 제거 (구조적 특성으로 대체)

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

# XGBoost 모델 파라미터
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': ['mae', 'mape'],
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'random_state': 42
}

# 모델 학습
model = xgb.XGBRegressor(**xgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
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
          f"Workers: {test_data['Number of Workers'].iloc[i]}")
    print(f"Predicted: {pred:.2f} MB/s, Actual: {actual:.2f} MB/s, Error: {error:.2f}%")

# 특성 중요도 분석
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# 모델 저장
model.save_model('comm_predictor_v2.json')
