import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 데이터 로드 (경로는 적절히 수정하세요)
df = pd.read_csv('../data/dataset_v2.csv')

# 모델 아키텍처 특성 정의 (원본 코드에서 가져옴)
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

# XGBoost 파라미터 (원본 코드에서 가져옴)
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

# 전체 데이터셋에서 사용되는 모든 고유한 Pattern과 Model 값을 미리 저장
unique_patterns = df['Pattern'].unique()
unique_models = df['Model'].unique()

def create_features(df, all_patterns=unique_patterns, all_models=unique_models):
    """
    특성 생성 함수 (원본 코드에서 가져옴, 일관된 특성 생성을 위해 수정됨)

    Args:
        df: 입력 데이터프레임
        all_patterns: 모든 가능한 Pattern 값 목록
        all_models: 모든 가능한 Model 값 목록
    """
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

    # 추가 모델 관련 특성
    features['num_parameters'] = df['Number of Parameters'].astype(float)
    features['num_layers'] = df['Number of Layers'].astype(float)

    # 통신/계산 특성
    features['compute_intensity'] = features['param_density'] * features['tensor_size']
    features['comm_intensity'] = features['tensor_size'] * N / features['batch_size']

    # 특성 상호작용 - 워커 수와의 관계에 더 집중
    features['ring_worker_interaction'] = features['ring_volume'] * np.log2(N)
    features['ps_worker_interaction'] = features['ps_volume'] * features['ps_worker_ratio']

    # 워커 수에 따른 확장성 포착을 위한 추가 특성
    features['worker_sqrt'] = np.sqrt(N)
    features['worker_squared'] = N**2
    features['worker_log2'] = np.log2(N)

    # 로그 변환 특성
    features['log_tensor_size'] = np.log1p(df['tensorsize'])
    features['log_batch_size'] = np.log(df['Batch Size'])

    # 데이터셋 크기
    features['dataset_size'] = df['Data Set'].map({'cifar10': 1, 'imagenet': 10, 'squad': 5})

    # Pattern을 범주형 변수로 변환 (원-핫 인코딩) - 모든 패턴에 대해 일관되게 열 생성
    for pattern in all_patterns:
        col_name = f'pattern_{pattern}'
        features[col_name] = (df['Pattern'] == pattern).astype(int)

    # 모델을 범주형 변수로 변환 (원-핫 인코딩) - 모든 모델에 대해 일관되게 열 생성
    for model_name in all_models:
        col_name = f'model_{model_name}'
        features[col_name] = (df['Model'] == model_name).astype(int)

    return features

# 특성 생성
X = create_features(df)
y = df['Sum of Max TX+RX (MB/s)']

# 데이터셋의 워커 수 확인
worker_counts = sorted(df['Number of Workers'].unique())
print(f"데이터셋의 워커 수: {worker_counts}")

# 워커 수에 기반한 교차 검증 수행
results = []

# 요구사항에 맞게 교차 검증 수행
# 1. 2, 8을 학습시키고 4의 예측 테스트
# 2. 4, 8을 학습시키고 2의 예측 테스트
# 3. 2, 4를 학습시키고 8의 예측 테스트
cv_combinations = [
    {'train': [2, 8], 'test': 4},
    {'train': [4, 8], 'test': 2},
    {'train': [2, 4], 'test': 8}
]

# 교차 검증 수행
for cv_setup in cv_combinations:
    train_workers = cv_setup['train']
    test_worker = cv_setup['test']

    # 훈련 및 테스트 마스크 생성
    train_mask = df['Number of Workers'].isin(train_workers)
    test_mask = df['Number of Workers'] == test_worker

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 모델 학습
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    # 테스트 예측
    test_pred = model.predict(X_test)

    # 평가 지표 계산
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

    # 데이터 패턴별로 분석
    patterns = df.loc[test_mask, 'Pattern'].unique()
    pattern_results = []

    for pattern in patterns:
        pattern_mask = (df['Pattern'] == pattern) & test_mask
        if sum(pattern_mask) > 0:
            pattern_y_test = y[pattern_mask]
            pattern_y_pred = model.predict(X[pattern_mask])
            pattern_mae = mean_absolute_error(pattern_y_test, pattern_y_pred)
            pattern_mape = mean_absolute_percentage_error(pattern_y_test, pattern_y_pred) * 100
            pattern_results.append({
                'pattern': pattern,
                'mae': pattern_mae,
                'mape': pattern_mape,
                'sample_count': sum(pattern_mask)
            })

    # 모델별로 분석
    models = df.loc[test_mask, 'Model'].unique()
    model_results = []

    for model_name in models:
        model_mask = (df['Model'] == model_name) & test_mask
        if sum(model_mask) > 0:
            model_y_test = y[model_mask]
            model_y_pred = model.predict(X[model_mask])
            model_mae = mean_absolute_error(model_y_test, model_y_pred)
            model_mape = mean_absolute_percentage_error(model_y_test, model_y_pred) * 100
            model_results.append({
                'model': model_name,
                'mae': model_mae,
                'mape': model_mape,
                'sample_count': sum(model_mask)
            })

    # 결과 저장
    results.append({
        'train_workers': train_workers,
        'test_worker': test_worker,
        'train_samples': sum(train_mask),
        'test_samples': sum(test_mask),
        'mae': test_mae,
        'mape': test_mape,
        'pattern_results': pattern_results,
        'model_results': model_results
    })

# 결과 출력
print("\n===== Leave-One-Worker-Count-Out Evaluation Summary =====")
for result in results:
    print(f"\nTrained on {result['train_workers']} workers, tested on {result['test_worker']} workers:")
    print(f"  Overall MAE: {result['mae']:.2f}, MAPE: {result['mape']:.2f}%")
    print(f"  Train samples: {result['train_samples']}, Test samples: {result['test_samples']}")

    print("\n  Pattern-specific results:")
    for pattern_result in result['pattern_results']:
        print(f"    {pattern_result['pattern']}: MAE={pattern_result['mae']:.2f}, MAPE={pattern_result['mape']:.2f}%, Samples={pattern_result['sample_count']}")

    print("\n  Model-specific results:")
    for model_result in result['model_results']:
        print(f"    {model_result['model']}: MAE={model_result['mae']:.2f}, MAPE={model_result['mape']:.2f}%, Samples={model_result['sample_count']}")

# 특성 중요도 분석
def train_global_model():
    # 전체 데이터셋으로 모델 학습
    global_model = xgb.XGBRegressor(**xgb_params)
    global_model.fit(X, y)

    # 특성 중요도 계산
    importances = global_model.feature_importances_
    feature_names = X.columns

    # 특성 중요도 정렬
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:15]]

    print("\n===== Top 15 Feature Importances =====")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")

    return global_model, top_features

# 전역 모델 학습 및 특성 중요도 분석
global_model, top_features = train_global_model()

# 16개, 32개 워커에 대한 예측 함수
def predict_higher_worker_counts(model, base_df, target_worker_counts):
    """
    훈련된 모델을 사용하여 더 높은 워커 수에 대한 예측 수행

    Args:
        model: 훈련된 XGBoost 모델
        base_df: 기본 데이터프레임
        target_worker_counts: 예측할 워커 수 리스트 (예: [16, 32])

    Returns:
        예측 결과 데이터프레임
    """
    # 예측 결과를 저장할 리스트
    predictions = []

    # 각 모델, 패턴 조합에 대해 예측 수행
    for model_name in unique_models:
        for pattern in unique_patterns:
            # 원래 8-워커 데이터 가져오기
            original_data = base_df[(base_df['Model'] == model_name) &
                                  (base_df['Pattern'] == pattern) &
                                  (base_df['Number of Workers'] == 8)]

            if len(original_data) == 0:
                continue

            # 원본 행 복사
            base_row = original_data.iloc[0].copy()

            for worker_count in target_worker_counts:
                # 새 행 생성
                new_row = base_row.copy()
                new_row['Number of Workers'] = worker_count

                # Batch Size 스케일링 (워커 수 비율에 맞게)
                new_row['Batch Size'] = base_row['Batch Size'] * (worker_count / 8)

                # Parameter Server 수 조정 (패턴이 parameterserver인 경우)
                if pattern == 'parameterserver':
                    new_row['Number of PSs'] = worker_count  # 일반적으로 워커 수와 동일하게 설정

                # 특성 생성 - 전체 훈련 데이터셋과 동일한 특성 집합 생성
                features_df = pd.DataFrame([new_row])
                features = create_features(features_df, unique_patterns, unique_models)

                # 모델 예측
                predicted_bandwidth = model.predict(features)[0]

                # 결과 저장
                predictions.append({
                    'Model': model_name,
                    'Pattern': pattern,
                    'Number of Workers': worker_count,
                    'Batch Size': new_row['Batch Size'],
                    'Number of PSs': new_row['Number of PSs'],
                    'Predicted Bandwidth (MB/s)': predicted_bandwidth
                })

    # 결과 데이터프레임 반환
    return pd.DataFrame(predictions)

# 16, 32개 워커에 대한 예측 수행
higher_worker_predictions = predict_higher_worker_counts(global_model, df, [16, 32])

# 결과 출력
print("\n===== Predictions for Higher Worker Counts =====")
print(higher_worker_predictions)

# 워커 수에 따른 확장성 분석 시각화 함수
def visualize_scaling(df, predictions, model_name='resnet50'):
    """
    특정 모델의 워커 수에 따른 확장성 시각화

    Args:
        df: 원본 데이터프레임
        predictions: 예측 결과 데이터프레임
        model_name: 시각화할 모델 이름
    """
    plt.figure(figsize=(12, 8))

    # 패턴별로 그래프 그리기
    for pattern in df['Pattern'].unique():
        # 실제 데이터 필터링
        actual_data = df[(df['Model'] == model_name) & (df['Pattern'] == pattern)]
        workers = actual_data['Number of Workers'].values
        bandwidth = actual_data['Sum of Max TX+RX (MB/s)'].values

        # 실제 데이터 플롯
        plt.scatter(workers, bandwidth, label=f'{pattern} (Actual)',
                   marker='o', s=100)

        # 선 그래프 추가 (실제 데이터)
        if len(workers) > 1:
            plt.plot(workers, bandwidth, '--', alpha=0.7)

        # 예측 데이터 필터링
        pred_data = predictions[(predictions['Model'] == model_name) &
                               (predictions['Pattern'] == pattern)]

        if len(pred_data) > 0:
            pred_workers = pred_data['Number of Workers'].values
            pred_bandwidth = pred_data['Predicted Bandwidth (MB/s)'].values

            # 모든 데이터 포인트 합치기 (실제 + 예측)
            all_workers = np.concatenate([workers, pred_workers])
            all_bandwidth = np.concatenate([bandwidth, pred_bandwidth])
            sorted_indices = np.argsort(all_workers)
            all_workers = all_workers[sorted_indices]
            all_bandwidth = all_bandwidth[sorted_indices]

            # 예측 데이터 플롯
            plt.scatter(pred_workers, pred_bandwidth,
                       label=f'{pattern} (Predicted)', marker='x', s=100)

            # 연결선 그래프 (모든 포인트)
            plt.plot(all_workers, all_bandwidth, '-', alpha=0.5)

    plt.xlabel('Number of Workers (GPUs)', fontsize=14)
    plt.ylabel('Bandwidth (MB/s)', fontsize=14)
    plt.title(f'Scaling Analysis for {model_name}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # x축 눈금 명시적 설정
    plt.xticks([2, 4, 8, 16, 32])

    plt.tight_layout()
    plt.savefig(f'{model_name}_scaling_analysis.png')
    plt.close()

# 몇 가지 대표 모델에 대해 확장성 시각화
for model_name in ['resnet50', 'vgg16', 'bert', 'alexnet']:
    visualize_scaling(df, higher_worker_predictions, model_name)

print("\nScaling analysis visualizations have been saved as PNG files.")
