# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:42:04 2025

@author: km000009
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.inspection import permutation_importance
import joblib
import time
import os
import matplotlib.pyplot as plt
import psutil
import logging
from datetime import datetime
import sys
import contextlib

def get_memory_usage():
    process = psutil.Process(os.getpid())
    gb = process.memory_info().rss / 1024 / 1024 / 1024
    return f"{gb:.2f}GB"

def setup_logging(pathC):
    log_file = os.path.join(pathC, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def convert(seconds):
    days = seconds // 86400
    time_part = time.strftime("%H:%M:%S", time.gmtime(seconds % 86400))
    return f"{days}d {time_part}" if days > 0 else time_part

def split_data(data):
    train, test_and_validate = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])
    test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['target'])
    return train, validate, test

def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    class_weights = {cls: np.sqrt(max(counts) / count) for cls, count in zip(classes, counts)}
    sample_weights = np.array([class_weights[label] for label in y])
    return sample_weights

def train_and_evaluate(X_train, y_train, X_valid, y_validate, X_test, y_test, pathC, model_name, report_name, feature_names, feature_importances):
    import time
    start = time.time()
    sample_weights = compute_class_weights(y_train)
    model = lgb.LGBMClassifier(
        random_state=42,
        objective='multiclass',
        n_jobs=-1,
        force_row_wise=True,
        verbose=-1,
        verbosity=-1
    )
    # Suppress LightGBM stdout/stderr during fit
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid, y_validate)]
        )
    best_params = model.get_params()
    joblib.dump(model, os.path.join(pathC, model_name))
    y_pred_train = model.predict(X_train).astype(int)
    y_pred_valid = model.predict(X_valid).astype(int)
    y_pred_test = model.predict(X_test).astype(int)
    accuracy_train = round(accuracy_score(y_train, y_pred_train), 4)
    accuracy_valid = round(accuracy_score(y_validate, y_pred_valid), 4)
    accuracy_test = round(accuracy_score(y_test, y_pred_test), 4)
    f1_train = round(f1_score(y_train, y_pred_train, average='weighted'), 4)
    f1_valid = round(f1_score(y_validate, y_pred_valid, average='weighted'), 4)
    f1_test = round(f1_score(y_test, y_pred_test, average='weighted'), 4)
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_valid = confusion_matrix(y_validate, y_pred_valid)
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(f"Features: {list(feature_names)}")
    print(f"Training accuracy: {accuracy_train}")
    print(f"Validation accuracy: {accuracy_valid}")
    print(f"Test accuracy: {accuracy_test}\n")
    with open(os.path.join(pathC, report_name), 'w') as f:
        f.write(f"Top features: {list(feature_names)}\n")
        f.write(f"Feature importances: {list(feature_importances)}\n")
        f.write(f"Best hyperparameters: {best_params}\n")
        f.write(f"Training accuracy: {accuracy_train}\n")
        f.write(f"Validation accuracy: {accuracy_valid}\n")
        f.write(f"Test accuracy: {accuracy_test}\n")
        f.write(f"Training F1 (weighted): {f1_train}\n")
        f.write(f"Validation F1 (weighted): {f1_valid}\n")
        f.write(f"Test F1 (weighted): {f1_test}\n")
        f.write(f"Confusion Matrix (Train):\n{cm_train}\n")
        f.write(f"Confusion Matrix (Validation):\n{cm_valid}\n")
        f.write(f"Confusion Matrix (Test):\n{cm_test}\n")
    end = time.time()
    time_taken = round(end - start, 2)
    return {
        'num_features': len(feature_names),
        'features': list(feature_names),
        'accuracy_train': accuracy_train,
        'accuracy_valid': accuracy_valid,
        'accuracy_test': accuracy_test,
        'f1_train': f1_train,
        'f1_valid': f1_valid,
        'f1_test': f1_test,
        'time_taken': time_taken
    }

if __name__ == "__main__":
    start = time.time()
    pathC = os.environ.get('CLASSIFICATION_DIR', '/dbfs/mnt/lab/unrestricted/KritiM/classification/')
    training_file = os.environ.get('TRAINING_FILE', os.path.join(pathC, 'trainingSample.csv'))
    os.makedirs(pathC, exist_ok=True)
    log_file = setup_logging(pathC)
    logging.info(f"Starting training script - Output directory: {pathC}")
    import sklearn
    logging.info(f"Python packages: lightgbm {lgb.__version__}, sklearn {sklearn.__version__}, pandas {pd.__version__}")
    logging.info(f"Initial memory usage: {get_memory_usage()}")
    logging.info('Loading the labelled data...')
    try:
        df = pd.read_csv(training_file)
        logging.info(f"Successfully loaded data from {training_file}")
        logging.info(f"Initial dataframe shape: {df.shape}")
        logging.info(f"Memory usage after load: {get_memory_usage()}")
        df = df.drop_duplicates()
        df = df.dropna()
        logging.info(f"Shape after cleaning: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading/cleaning data: {str(e)}")
        raise
    df['target'] = df['target'].astype(int) - 1
    print("Class distribution:")
    print(df['target'].value_counts(normalize=True))
    print('assign categorical and numerical columns...')
    categorical_cols = ['Landcover_LE', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                        'Aggregate_texture', 'Aquifers', 'bedrock_raster_50m', 'ALC_old']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    for col in df.select_dtypes(include='number').columns:
        if col != 'target':
            df[col] = pd.to_numeric(df[col], downcast='float')
    print(f"Shape of training data: {df.shape}")
    print(f"Unique target values: {np.unique(df['target'])}")
    train, validate, test = split_data(df)
    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_valid = validate.drop('target', axis=1)
    y_validate = validate['target']
    X_test = test.drop('target', axis=1)
    y_test = test['target']
    num_cols = [col for col in X_train.columns if col not in categorical_cols]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    joblib.dump(scaler, os.path.join(pathC, 'scaler.joblib'))
    # --- PERMUTATION FEATURE IMPORTANCE FEATURE SELECTION ---
    # Train initial model to get feature importances
    initial_model = xgb.XGBClassifier(
        random_state=42, enable_categorical=True, objective='multi:softmax',
        num_class=len(np.unique(y_train)), eval_metric='mlogloss', tree_method='hist'
    )
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    }
    random_search = RandomizedSearchCV(
        initial_model, param_grid, n_iter=20, cv=3, scoring='f1_weighted',
        n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train, sample_weight=compute_class_weights(y_train))
    initial_model = random_search.best_estimator_
    perm_importance = permutation_importance(initial_model, X_valid, y_validate, n_repeats=3, random_state=42, n_jobs=-1)
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    # Select features incrementally
    feature_counts = range(3, 15, 1)
    results = []
    for n in feature_counts:
        print(f"\nTraining with top {n} features...")
        # Select top n features based on permutation importance
        selected_features = feature_importance_df['Feature'].head(n).values
        selected_importances = feature_importance_df['Importance'].head(n).values
        # Subset data
        X_train_n = X_train[selected_features]
        X_valid_n = X_valid[selected_features]
        X_test_n = X_test[selected_features]
        model_name = f'model_{n}_features.joblib'
        report_name = f'report_{n}_features.txt'
        print(f"\nTraining and evaluating with top {n} features: {list(selected_features)}")
        res = train_and_evaluate(
            X_train_n, y_train, X_valid_n, y_validate, X_test_n, y_test, pathC,
            model_name, report_name, selected_features, selected_importances
        )
        results.append(res)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(pathC, 'feature_selection_results.csv'), index=False)
    print(f"\nResults table saved to {os.path.join(pathC, 'feature_selection_results.csv')}")
    end = time.time()
    time_taken = convert(end-start)
    logging.info(f"Total processing time: {time_taken}")
    logging.info(f"Final memory usage: {get_memory_usage()}")
    logging.info(f"Log file saved to: {log_file}")
    print(f"\nScript completed successfully. Check the log file at: {log_file}")