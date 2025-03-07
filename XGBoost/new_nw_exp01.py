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

# Create features
X = create_simplified_features(df)
y = df['Sum of Max TX+RX (MB/s)']

# Get unique worker counts and model names
unique_workers = df['Number of Workers'].unique()
unique_models = df['Model'].unique()
print(f"\nUnique worker counts in dataset: {unique_workers}")
print(f"Unique models in dataset: {unique_models}")

# Leave-One-Worker-Out evaluation for each model
results = []
print("\n===== Training and Evaluating with Leave-One-Worker-Out Strategy for Each Model =====")
print(f"Using features: {X.columns.tolist()}")

# Create a list of (model, worker) combinations
model_worker_combinations = []
for model in unique_models:
    for worker in unique_workers:
        if len(df[(df['Model'] == model) & (df['Number of Workers'] == worker)]) > 0:
            model_worker_combinations.append((model, worker))

print(f"\nFound {len(model_worker_combinations)} valid (model, worker) combinations for testing")

# Perform leave-one-out analysis for each (model, worker) combination
for model_name, left_out_worker in model_worker_combinations:
    # Test mask: samples with the current model and worker count
    test_mask = (df['Model'] == model_name) & (df['Number of Workers'] == left_out_worker)

    # Skip if no test samples
    if test_mask.sum() == 0:
        print(f"Skipping {model_name} with worker {left_out_worker}: No test samples")
        continue

    # Train mask: samples with the same model but different worker counts
    train_mask = (df['Model'] == model_name) & (df['Number of Workers'] != left_out_worker)

    # Skip if no train samples
    if train_mask.sum() == 0:
        print(f"Skipping {model_name} with worker {left_out_worker}: No train samples")
        continue

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_workers = df[train_mask]['Number of Workers'].unique()
    print(f"\nModel: {model_name}, Training on workers {train_workers}, testing on worker {left_out_worker}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train model
    model = xgb.XGBRegressor(early_stopping_rounds=30, **xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    # Predict and evaluate
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

    # Store results
    results.append({
        'model': model_name,
        'left_out_worker': left_out_worker,
        'train_workers': ', '.join(map(str, train_workers)),
        'test_mae': test_mae,
        'test_mape': test_mape
    })

# Results summary
results_df = pd.DataFrame(results)
print("\n===== Leave-One-Worker-Out Evaluation Summary =====")
print(results_df)

# Calculate average metrics by model
model_avg = results_df.groupby('model')[['test_mae', 'test_mape']].mean()
print("\nAverage performance by model across all worker counts:")
print(model_avg.sort_values('test_mae'))

# Calculate average metrics by worker count
worker_avg = results_df.groupby('left_out_worker')[['test_mae', 'test_mape']].mean()
print("\nAverage performance by worker count:")
print(worker_avg)

# Visualization
plt.figure(figsize=(15, 10))

# Model performance comparison
plt.subplot(2, 1, 1)
model_pivot = results_df.pivot(index='model', columns='left_out_worker', values='test_mae')
model_pivot.plot(kind='bar', figsize=(15, 7))
plt.title('MAE by Model and Worker Count')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Model')
plt.legend(title='Left-out Worker')
plt.xticks(rotation=45)

# Worker performance comparison
plt.subplot(2, 1, 2)
worker_pivot = results_df.pivot(index='left_out_worker', columns='model', values='test_mae')
worker_pivot.plot(kind='bar', figsize=(15, 7))
plt.title('MAE by Worker Count and Model')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Worker Count')
plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('model_worker_performance.png')
print("\nVisualization saved as 'model_worker_performance.png'")

# Calculate feature importance across all models
print("\n===== Overall Feature Importance =====")
# Train a model on all data
full_model = xgb.XGBRegressor(**xgb_params)
full_model.fit(X, y)

# Print overall feature importance
importance = full_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("Overall feature importance across all data:")
for i, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Feature importance for each model type (averaged across worker counts)
print("\n===== Feature Importance by Model Type =====")
for model_name in unique_models:
    print(f"\nFeature importance for {model_name}:")

    # Filter data for this model
    model_mask = df['Model'] == model_name
    if model_mask.sum() == 0:
        print(f"No data for {model_name}")
        continue

    model_X = X[model_mask]
    model_y = y[model_mask]

    if len(model_X) < 10:  # Require minimum number of samples
        print(f"Too few samples for {model_name} to calculate reliable feature importance")
        continue

    # Train model on all data for this model type
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(model_X, model_y)

    # Print feature importance
    model_importance = model.feature_importances_
    model_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model_importance})
    model_importance_df = model_importance_df.sort_values('Importance', ascending=False)

    for i, row in model_importance_df.iterrows():
        if row['Importance'] > 0.01:  # Only show features with non-trivial importance
            print(f"{row['Feature']}: {row['Importance']:.4f}")
