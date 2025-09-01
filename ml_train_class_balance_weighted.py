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
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import time
import os
import matplotlib.pyplot as plt

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

def train_and_evaluate(X_train, y_train, X_valid, y_validate, X_test, y_test, pathC, model_name, report, num_features):
    start = time.time()
    
    # Compute class weights
    sample_weights = compute_class_weights(y_train)
    
    # Initialize XGBoost classifier for hyperparameter tuning
    base_model = xgb.XGBClassifier(
        random_state=42,
        enable_categorical=True,
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        tree_method='hist'
    )
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5],
        'subsample': [0.5, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1.0],
        'min_child_weight': [1, 5, 10, 20],
        'reg_lambda': [0, 1, 10, 100], 
        'reg_alpha': [0, 0.1, 1, 10],        
    }
    
    # Perform hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=100,  # Reduced for speed
        cv=5,       # Reduced for speed
        scoring='f1_weighted',
        n_jobs=-1,  # Parallel processing
        verbose=1,
        random_state=42,
        error_score='raise'
    )
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Best parameters
    best_params = random_search.best_params_
    print(f"Best parameters for {num_features} features: {best_params}")
    
    # Train final model with native XGBoost for early stopping
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_validate, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    
    model = xgb.train(
        params={**best_params, 'objective': 'multi:softmax', 'num_class': len(np.unique(y_train)), 'eval_metric': 'mlogloss', 'tree_method': 'hist'},
        dtrain=dtrain,
        num_boost_round=best_params['n_estimators'],
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # Save the model
    joblib.dump(model, os.path.join(pathC, model_name))
    print(f"Model with {num_features} features saved to {os.path.join(pathC, model_name)}")
    
    # Feature importances (permutation importance using scikit-learn wrapper for consistency)
    sk_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        enable_categorical=True,
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        tree_method='hist'
    )
    sk_model.fit(X_train, y_train, sample_weight=sample_weights)  # Fit for permutation importance
    perm_importance = permutation_importance(sk_model, X_valid, y_validate, n_repeats=5, random_state=42, n_jobs=-1)
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    print(f"Permutation Feature Importances for {num_features} features:")
    print(feature_importance_df)
    
    # Evaluate on training data
    y_pred_train = model.predict(dtrain).astype(int)
    accuracy_train = round(accuracy_score(y_train, y_pred_train), 2)
    f1_train = round(f1_score(y_train, y_pred_train, average='weighted'), 2)
    print(f"\nAccuracy on training data ({num_features} features): {accuracy_train * 100:.2f}%")
    print(f"F1-Score (weighted) on training data: {f1_train * 100:.2f}%")
    print("Confusion Matrix (Train):")
    print(confusion_matrix(y_train, y_pred_train))
    print("Classification Report (Train):")
    print(classification_report(y_train, y_pred_train))
    
    # Evaluate on validation data
    y_pred_valid = model.predict(dvalid).astype(int)
    accuracy_valid = round(accuracy_score(y_validate, y_pred_valid), 2)
    f1_valid = round(f1_score(y_validate, y_pred_valid, average='weighted'), 2)
    print(f"Accuracy on validation data ({num_features} features): {accuracy_valid * 100:.2f}%")
    print(f"F1-Score (weighted) on validation data: {f1_valid * 100:.2f}%")
    print("Confusion Matrix (Validation):")
    print(confusion_matrix(y_validate, y_pred_valid))
    print("Classification Report (Validation):")
    print(classification_report(y_validate, y_pred_valid))
    
    # Evaluate on test data
    y_pred_test = model.predict(dtest).astype(int)
    accuracy_test = round(accuracy_score(y_test, y_pred_test), 2)
    f1_test = round(f1_score(y_test, y_pred_test, average='weighted'), 2)
    print(f"Accuracy on test data ({num_features} features): {accuracy_test * 100:.2f}%")
    print(f"F1-Score (weighted) on test data: {f1_test * 100:.2f}%")
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_pred_test))
    print("Classification Report (Test):")
    print(classification_report(y_test, y_pred_test))
    
    # Write report
    with open(os.path.join(pathC, report), "a") as f:
        f.write(f"\n\nResults for {num_features} features:\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write("Permutation Feature Importances:\n")
        f.write(feature_importance_df.to_string())
        f.write(f"\nAccuracy on training data: {accuracy_train}")
        f.write(f"\nF1-Score (weighted) on training data: {f1_train}")
        f.write(f"\nAccuracy on validation data: {accuracy_valid}")
        f.write(f"\nF1-Score (weighted) on validation data: {f1_valid}")
        f.write(f"\nAccuracy on test data: {accuracy_test}")
        f.write(f"\nF1-Score (weighted) on test data: {f1_test}\n")
    
    end = time.time()
    print(f"Time taken for {num_features} features: {convert(end - start)}")
    
    return accuracy_train, accuracy_valid, accuracy_test, f1_train, f1_valid, f1_test

def plot_metrics(feature_counts, accuracies_train, accuracies_valid, accuracies_test, 
                f1_train, f1_valid, f1_test, pathC):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.plot(feature_counts, accuracies_train, label='Training Accuracy', marker='o', color='#1f77b4')
    plt.plot(feature_counts, accuracies_valid, label='Validation Accuracy', marker='o', color='#ff7f0e')
    plt.plot(feature_counts, accuracies_test, label='Test Accuracy', marker='o', color='#2ca02c')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Features')
    plt.legend()
    plt.grid(True)
    
    # Plot F1-scores
    plt.subplot(1, 2, 2)
    plt.plot(feature_counts, f1_train, label='Training F1-Score', marker='o', color='#1f77b4')
    plt.plot(feature_counts, f1_valid, label='Validation F1-Score', marker='o', color='#ff7f0e')
    plt.plot(feature_counts, f1_test, label='Test F1-Score', marker='o', color='#2ca02c')
    plt.xlabel('Number of Features')
    plt.ylabel('F1-Score (Weighted)')
    plt.title('F1-Score vs. Number of Features')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pathC, 'metrics_vs_features.png'))
    plt.close()
    print(f"Metrics plot saved to {os.path.join(pathC, 'metrics_vs_features.png')}")

if __name__ == "__main__":
    start = time.time()
    pathC = 'classification/'  # Adjust path as needed
    training_file = os.path.join(pathC, 'trainingSample.csv')
    os.makedirs(pathC, exist_ok=True)
    
    # Load data
    print('load the labelled data....')
    df = pd.read_csv(training_file)
    # df = df.drop(columns=['Land_cover'], errors='ignore')
    df = df.dropna()
    
    # Prepare target and features
    df['target'] = df['target'].astype(int) - 1  # Shift to zero-based
    print("Class distribution:")
    print(df['target'].value_counts(normalize=True))
    
    print('assign categorical and numerical columns...')
    categorical_cols = ['Land_cover', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                        'Aggregate_texture', 'Peat', 'Aquifers']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Downcast numerical columns
    for col in df.select_dtypes(include='number').columns:
        if col != 'target':
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    print(f"Shape of training data: {df.shape}")
    print(f"Unique target values: {np.unique(df['target'])}")
    
    # Split data
    train, validate, test = split_data(df)
    
    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_valid = validate.drop('target', axis=1)
    y_validate = validate['target']
    X_test = test.drop('target', axis=1)
    y_test = test['target']
    
    # Scale numerical columns
    num_cols = [col for col in X_train.columns if col not in categorical_cols]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save scaler
    joblib.dump(scaler, os.path.join(pathC, 'scaler.joblib'))
    
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
    feature_counts = range(3, len(X_train.columns) + 1, 2)
    results = []
    
    for k in feature_counts:
        print(f"\nTraining with top {k} features...")
        # Select top k features based on permutation importance
        selected_features = feature_importance_df['Feature'].head(k).values
        
        # Subset data
        X_train_subset = X_train[selected_features]
        X_valid_subset = X_valid[selected_features]
        X_test_subset = X_test[selected_features]
        
        # Train and evaluate
        model_name = f'model_{k}_features.joblib'
        report = f'report_{k}_features.txt'
        acc_train, acc_valid, acc_test, f1_train, f1_valid, f1_test = train_and_evaluate(
            X_train_subset, y_train, X_valid_subset, y_validate, X_test_subset, y_test, 
            pathC, model_name, report, k
        )
        
        results.append({
            'num_features': k,
            'acc_train': acc_train,
            'acc_valid': acc_valid,
            'acc_test': acc_test,
            'f1_train': f1_train,
            'f1_valid': f1_valid,
            'f1_test': f1_test
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(pathC, 'metrics_vs_features.csv'), index=False)
    print(f"Results saved to {os.path.join(pathC, 'metrics_vs_features.csv')}")
    
    # Plot metrics
    plot_metrics(
        results_df['num_features'],
        results_df['acc_train'],
        results_df['acc_valid'],
        results_df['acc_test'],
        results_df['f1_train'],
        results_df['f1_valid'],
        results_df['f1_test'],
        pathC
    )
    end = time.time()
    time_taken = convert(end-start)
    print(f'time taken for processing: {time_taken}')
