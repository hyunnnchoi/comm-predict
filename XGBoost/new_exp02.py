import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# XGBoost 파라미터 (원본에서 유지)
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

def create_simplified_features(df):
    """요청된 대로 단순화된 특성 세트를 생성합니다"""
    features = pd.DataFrame()

    # 기본 특성
    features['tensor_size'] = df['tensorsize']
    features['num_workers'] = df['Number of Workers']
    features['batch_size'] = df['Batch Size']
    features['num_ps'] = df['Number of PSs']

    # PS/AR 여부 추가 (중요한 특성)
    # 파라미터 서버가 0개이면 AR(AllReduce), 그렇지 않으면 PS
    features['is_allreduce'] = (df['Number of PSs'] == 0).astype(int)

    # 모델 구조 특성
    # 참고: model_depth는 원래 사전에서 조회했지만 여기서는 직접 매핑합니다
    # model_depth를 위한 간단한 매핑 정의
    model_depth_map = {
        'alexnet': 8,
        'resnet110': 110,
        'resnet44': 44,
        'resnet56': 56,
        'densenet40_k12': 40,
        'densenet100_k12': 100,
        'googlenet': 22,
        'vgg16': 16,
        'inception3': 48,
        'resnet50': 50,
        'bert': 12,
        'gpt2': 12
    }

    features['model_depth'] = df['Model'].map(model_depth_map)

    # 이 특성들은 데이터셋에서 직접 가져옵니다
    features['num_parameters'] = df['Number of Parameters'].astype(float)
    features['num_layers'] = df['Number of Layers'].astype(float)

    return features

# 데이터 로드 및 전처리
# CSV 파일을 명시적으로 로드하고 열 이름 보존
df = pd.read_csv('../data/dataset_v2.csv')

# 열 이름에 공백이 있는지 확인
print("원본 열 이름:", df.columns.tolist())

# 열 이름에 있을 수 있는 공백 제거
df.columns = df.columns.str.strip()
print("정리된 열 이름:", df.columns.tolist())

# 필요한 열이 존재하는지 확인
required_columns = ['tensorsize', 'Number of Workers', 'Batch Size', 'Number of PSs',
                   'Model', 'Number of Parameters', 'Number of Layers']

for col in required_columns:
    if col not in df.columns:
        print(f"경고: 필요한 열 '{col}'이 데이터셋에 없습니다!")

# 특성 생성 및 모델 학습
X = create_simplified_features(df)
y = df['Sum of Max TX+RX (MB/s)']

# Leave-One-Model-Out 평가 (한 모델을 제외하고 학습)
results = []
unique_models = df['Model'].unique()

print("\n===== 단순화된 특성 세트로 학습 및 평가 중 =====")
print(f"다음 특성만 사용: {X.columns.tolist()}")

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
print("\n===== Leave-One-Model-Out 평가 요약 =====")
print(results_df)

# 평균 메트릭 계산
avg_mae = results_df['test_mae'].mean()
avg_mape = results_df['test_mape'].mean()
print(f"\n모든 모델에 대한 평균 MAE: {avg_mae:.2f}")
print(f"모든 모델에 대한 평균 MAPE: {avg_mape:.2f}%")

# 선택 사항: 특성 중요도 플롯
if len(unique_models) > 0:
    # 특성 중요도를 얻기 위해 모든 데이터로 모델 학습
    full_model = xgb.XGBRegressor(**xgb_params)
    full_model.fit(X, y)

    # 특성 중요도 플롯
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(full_model, max_num_features=7)
    plt.title('특성 중요도 - 단순화된 모델')
    plt.tight_layout()
    plt.savefig('feature_importance_simplified.png')
    print("\n특성 중요도 플롯이 'feature_importance_simplified.png'로 저장되었습니다")
