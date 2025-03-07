import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# XGBoost parameters (preserved from original)
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
    """Creates simplified feature set as requested"""
    features = pd.DataFrame()
    # Basic features
    features['tensor_size'] = df['tensorsize']
    features['num_workers'] = df['Number of Workers']
    features['batch_size'] = df['Batch Size']
    features['num_ps'] = df['Number of PSs']

    # Add PS/AR flag (important feature)
    # If parameter servers = 0, it's AR (AllReduce), otherwise PS
    features['is_allreduce'] = (df['Number of PSs'] == 0).astype(int)

    # Model architecture features
    # Note: model_depth was looked up in a dictionary originally, but here we map directly
    # Simple mapping for model_depth
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

    # These features are taken directly from the dataset
    features['num_parameters'] = df['Number of Parameters'].astype(float)
    features['num_layers'] = df['Number of Layers'].astype(float)

    return features

# Load and preprocess data
# Load CSV file explicitly and preserve column names
df = pd.read_csv('../data/dataset_v2.csv')

# Check if column names have spaces
print("Original column names:", df.columns.tolist())

# Remove any spaces that might be in column names
df.columns = df.columns.str.strip()
print("Cleaned column names:", df.columns.tolist())

# Check if required columns exist
required_columns = ['tensorsize', 'Number of Workers', 'Batch Size', 'Number of PSs',
                   'Model', 'Number of Parameters', 'Number of Layers']
for col in required_columns:
    if col not in df.columns:
        print(f"Warning: Required column '{col}' not found in dataset!")

# Create features and train model
X = create_simplified_features(df)
y = df['Sum of Max TX+RX (MB/s)']

# Leave-One-Model-Out evaluation (train excluding one model)
results = []
unique_models = df['Model'].unique()
print("\n===== Training and Evaluating with Simplified Feature Set =====")
print(f"Using only the following features: {X.columns.tolist()}")

for left_out_model in unique_models:
    test_mask = df['Model'] == left_out_model
    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = xgb.XGBRegressor(early_stopping_rounds=30, **xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

    # Store AllReduce ratio for this model's data
    ar_ratio = X_test['is_allreduce'].mean()

    results.append({
        'left_out_model': left_out_model,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'ar_ratio': ar_ratio
    })

# Results summary
results_df = pd.DataFrame(results)
print("\n===== Leave-One-Model-Out Evaluation Summary =====")
print(results_df)

# Calculate average metrics
avg_mae = results_df['test_mae'].mean()
avg_mape = results_df['test_mape'].mean()
print(f"\nAverage MAE across all models: {avg_mae:.2f}")
print(f"Average MAPE across all models: {avg_mape:.2f}%")

# Analysis by communication pattern (AllReduce vs Parameter Server)
print("\n===== Performance Analysis by Communication Pattern =====")

# 1. Analysis of communication patterns in the entire dataset
ar_mask = X['is_allreduce'] == 1
ps_mask = X['is_allreduce'] == 0

# Output ratio of AllReduce and PS for all models
ar_count = ar_mask.sum()
ps_count = ps_mask.sum()
total_count = len(X)
print(f"Overall dataset: AllReduce ratio: {ar_count/total_count:.2%} ({ar_count}/{total_count}), PS ratio: {ps_count/total_count:.2%} ({ps_count}/{total_count})")

# 2. Analysis of results by communication pattern for each model test
# Separate results for AllReduce and PS
ar_results = {}
ps_results = {}
combined_results = []

for left_out_model in unique_models:
    test_mask = df['Model'] == left_out_model
    X_test, y_test = X[test_mask], y[test_mask]

    # Train model (same as in the loop above)
    X_train, y_train = X[~test_mask], y[~test_mask]
    model = xgb.XGBRegressor(early_stopping_rounds=30, **xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    # Predict on the entire test set
    test_pred = model.predict(X_test)

    # Split test set by communication pattern
    ar_test_mask = X_test['is_allreduce'] == 1
    ps_test_mask = X_test['is_allreduce'] == 0

    # Measure performance for each communication pattern
    if ar_test_mask.any():
        ar_y_test = y_test[ar_test_mask]
        ar_pred = test_pred[ar_test_mask]
        ar_mae = mean_absolute_error(ar_y_test, ar_pred)
        ar_mape = mean_absolute_percentage_error(ar_y_test, ar_pred) * 100
        ar_count = ar_test_mask.sum()
    else:
        ar_mae = np.nan
        ar_mape = np.nan
        ar_count = 0

    if ps_test_mask.any():
        ps_y_test = y_test[ps_test_mask]
        ps_pred = test_pred[ps_test_mask]
        ps_mae = mean_absolute_error(ps_y_test, ps_pred)
        ps_mape = mean_absolute_percentage_error(ps_y_test, ps_pred) * 100
        ps_count = ps_test_mask.sum()
    else:
        ps_mae = np.nan
        ps_mape = np.nan
        ps_count = 0

    # Store performance by communication pattern for each model
    combined_results.append({
        'left_out_model': left_out_model,
        'ar_count': ar_count,
        'ps_count': ps_count,
        'ar_mae': ar_mae,
        'ar_mape': ar_mape,
        'ps_mae': ps_mae,
        'ps_mape': ps_mape
    })

# Output performance results by communication pattern
combined_results_df = pd.DataFrame(combined_results)
print("\nPerformance by communication pattern for each model:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(combined_results_df)

# Calculate average performance by communication pattern
ar_avg_mae = combined_results_df['ar_mae'].mean()
ar_avg_mape = combined_results_df['ar_mape'].mean()
ps_avg_mae = combined_results_df['ps_mae'].mean()
ps_avg_mape = combined_results_df['ps_mape'].mean()

print("\nAverage performance by communication pattern:")
print(f"AllReduce - Average MAE: {ar_avg_mae:.2f}, Average MAPE: {ar_avg_mape:.2f}%")
print(f"Parameter Server - Average MAE: {ps_avg_mae:.2f}, Average MAPE: {ps_avg_mape:.2f}%")

# Visualize performance by communication pattern
plt.figure(figsize=(15, 10))

# MAE comparison graph
plt.subplot(2, 1, 1)
models = combined_results_df['left_out_model'].tolist()
ar_mae_values = combined_results_df['ar_mae'].tolist()
ps_mae_values = combined_results_df['ps_mae'].tolist()

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, ar_mae_values, width, label='AllReduce MAE')
plt.bar(x + width/2, ps_mae_values, width, label='Parameter Server MAE')
plt.axhline(y=ar_avg_mae, color='b', linestyle='--', alpha=0.5, label=f'Avg AR MAE: {ar_avg_mae:.2f}')
plt.axhline(y=ps_avg_mae, color='orange', linestyle='--', alpha=0.5, label=f'Avg PS MAE: {ps_avg_mae:.2f}')

plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('MAE Comparison by Communication Pattern for Each Model')
plt.xticks(x, models, rotation=45)
plt.legend()

# MAPE comparison graph
plt.subplot(2, 1, 2)
ar_mape_values = combined_results_df['ar_mape'].tolist()
ps_mape_values = combined_results_df['ps_mape'].tolist()

plt.bar(x - width/2, ar_mape_values, width, label='AllReduce MAPE')
plt.bar(x + width/2, ps_mape_values, width, label='Parameter Server MAPE')
plt.axhline(y=ar_avg_mape, color='b', linestyle='--', alpha=0.5, label=f'Avg AR MAPE: {ar_avg_mape:.2f}%')
plt.axhline(y=ps_avg_mape, color='orange', linestyle='--', alpha=0.5, label=f'Avg PS MAPE: {ps_avg_mape:.2f}%')

plt.xlabel('Model')
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison by Communication Pattern for Each Model')
plt.xticks(x, models, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('communication_pattern_performance.png')
print("\nVisualization saved as 'communication_pattern_performance.png'")

# Optional: Feature importance output to console
if len(unique_models) > 0:
    # Train model on all data to get feature importance
    full_model = xgb.XGBRegressor(**xgb_params)
    full_model.fit(X, y)

    # Print feature importance to console
    importance = full_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    print("\n===== Feature Importance =====")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
