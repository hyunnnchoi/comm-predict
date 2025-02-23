import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('../data/cnn_network_summary.csv')

# Feature engineering
def create_features(df):
    features = pd.DataFrame()

    # 기본 특성
    features['tensor_size'] = df['tensorsize']
    features['num_workers'] = df['Number of Workers']
    features['batch_size'] = df['Batch Size']
    features['num_ps'] = df['Number of PSs']

    # 도메인 지식 기반 특성
    features['tensor_per_worker'] = df['tensorsize'] / df['Number of Workers']
    features['comm_intensity'] = df['Batch Size'] * df['tensorsize'] / df['Number of Workers']

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

    # 모델 타입 (One-hot encoding)
    model_dummies = pd.get_dummies(df['Model'], prefix='model')
    features = pd.concat([features, model_dummies], axis=1)

    return features

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
    'eval_metric': 'mae',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
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

# 특성 중요도 시각화
plt.figure(figsize=(12, 6))
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=True)

plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 모델 저장
model.save_model('comm_predictor.json')
