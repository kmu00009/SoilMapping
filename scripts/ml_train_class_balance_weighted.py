# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, log_loss
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import time
import os
import psutil
import logging
from datetime import datetime

ENSEMBLE_SEEDS = [42, 123, 2024, 7, 99]  # List of seeds for ensemble


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    gb = process.memory_info().rss / 1024 / 1024 / 1024
    return f"{gb:.2f}GB"

def setup_logging(pathC):
    """Setup logging to both file and console"""
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

def train_and_evaluate_ensemble(X_train, y_train, X_valid, y_validate, X_test, y_test, pathC, model_name, report, seeds):
    start = time.time()
    n_classes = len(np.unique(y_train))
    # Store predictions for ensemble
    valid_preds_proba = []
    test_preds_proba = []
    train_preds_proba = []
    best_params_list = []
    feature_importance_list = []
    for seed in seeds:
        # Compute class weights
        sample_weights = compute_class_weights(y_train)
        # Initialize XGBoost classifier for hyperparameter tuning
        base_model = xgb.XGBClassifier(
            random_state=seed,
            enable_categorical=True,
            objective='multi:softprob',  # Use softprob for probability output
            num_class=n_classes,
            eval_metric='mlogloss',
            tree_method='hist'
        )
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_child_weight': [7, 10, 15],
            'gamma': [1.0, 2.0, 3.0],
            'subsample': [0.5, 0.6, 0.7],
            'colsample_bytree': [0.4, 0.5, 0.6],
            'reg_lambda': [10, 50, 100],
            'reg_alpha': [1, 5, 10],
            'max_delta_step': [0, 1, 5]
        }
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=50,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0,
            random_state=seed,
            error_score='raise'
        )
        random_search.fit(X_train, y_train, sample_weight=sample_weights)
        best_params = random_search.best_params_
        best_params_list.append(best_params)
        # Train final model
        model = xgb.XGBClassifier(
            **best_params,
            random_state=seed,
            enable_categorical=True,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            tree_method='hist'
        )
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_valid, y_validate)], early_stopping_rounds=30, verbose=False)
        # Save the model
        joblib.dump(model, os.path.join(pathC, f"{model_name}_seed{seed}.joblib"))
        # Feature importances
        perm_importance = permutation_importance(model, X_valid, y_validate, n_repeats=5, random_state=seed, n_jobs=-1)
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False)
        feature_importance_list.append(feature_importance_df)
        # Store probability predictions
        train_preds_proba.append(model.predict_proba(X_train))
        valid_preds_proba.append(model.predict_proba(X_valid))
        test_preds_proba.append(model.predict_proba(X_test))
    # Ensemble: average probabilities
    train_proba_avg = np.mean(train_preds_proba, axis=0)
    valid_proba_avg = np.mean(valid_preds_proba, axis=0)
    test_proba_avg = np.mean(test_preds_proba, axis=0)
    # Final predictions
    y_pred_train = np.argmax(train_proba_avg, axis=1)
    y_pred_valid = np.argmax(valid_proba_avg, axis=1)
    y_pred_test = np.argmax(test_proba_avg, axis=1)
    # Metrics
    accuracy_train = round(accuracy_score(y_train, y_pred_train), 2)
    f1_train = round(f1_score(y_train, y_pred_train, average='weighted'), 2)
    accuracy_valid = round(accuracy_score(y_validate, y_pred_valid), 2)
    f1_valid = round(f1_score(y_validate, y_pred_valid, average='weighted'), 2)
    accuracy_test = round(accuracy_score(y_test, y_pred_test), 2)
    f1_test = round(f1_score(y_test, y_pred_test, average='weighted'), 2)
    # Confusion matrices and reports
    cm_train = confusion_matrix(y_train, y_pred_train)
    cr_train = classification_report(y_train, y_pred_train)
    cm_valid = confusion_matrix(y_validate, y_pred_valid)
    cr_valid = classification_report(y_validate, y_pred_valid)
    cm_test = confusion_matrix(y_test, y_pred_test)
    cr_test = classification_report(y_test, y_pred_test)
    # Write report
    report_content = f"""
Ensemble Results for 11 features:
Best parameters (per seed): {best_params_list}
Permutation Feature Importances (first seed):
{feature_importance_list[0].to_string()}
Accuracy on training data: {accuracy_train}
F1-Score (weighted) on training data: {f1_train}
Confusion Matrix (Train):
{np.array2string(cm_train, separator=', ')}
Classification Report (Train):
{cr_train}
Accuracy on validation data: {accuracy_valid}
F1-Score (weighted) on validation data: {f1_valid}
Confusion Matrix (Validation):
{np.array2string(cm_valid, separator=', ')}
Classification Report (Validation):
{cr_valid}
Accuracy on test data: {accuracy_test}
F1-Score (weighted) on test data: {f1_test}
Confusion Matrix (Test):
{np.array2string(cm_test, separator=', ')}
Classification Report (Test):
{cr_test}
"""
    try:
        if os.path.exists(os.path.join(pathC, report)):
            with open(os.path.join(pathC, report), 'r') as f:
                existing_content = f.read()
        else:
            existing_content = ""
    except:
        existing_content = ""
    try:
        with open(os.path.join(pathC, report), 'w') as f:
            f.write(existing_content + report_content)
        logging.info(f"Report updated: {os.path.join(pathC, report)}")
    except Exception as e:
        logging.error(f"Could not write to {report}. Error: {str(e)}")
        fallback_path = os.path.join('/dbfs/FileStore/SoilMapping/reports/', report)
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path, 'w') as f:
            f.write(existing_content + report_content)
        logging.info(f"Report written to fallback location: {fallback_path}")
    end = time.time()
    print(f"Time taken for 11 features (ensemble): {convert(end - start)}")
    return accuracy_train, accuracy_valid, accuracy_test, f1_train, f1_valid, f1_test

if __name__ == "__main__":
    start = time.time()
    pathC = os.environ.get('CLASSIFICATION_DIR', '/dbfs/mnt/lab/unrestricted/KritiM/classification/')
    training_file = os.environ.get('TRAINING_FILE', os.path.join(pathC, 'trainingSample.csv'))
    os.makedirs(pathC, exist_ok=True)
    log_file = setup_logging(pathC)
    logging.info(f"Starting training script - Output directory: {pathC}")
    import sklearn
    logging.info(f"Python packages: xgboost {xgb.__version__}, sklearn {sklearn.__version__}, pandas {pd.__version__}")
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
    initial_model = xgb.XGBClassifier(
        random_state=42, enable_categorical=True, objective='multi:softprob',
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
    # Use only the 11 highest ranking features
    selected_features = feature_importance_df['Feature'].head(11).values
    X_train_subset = X_train[selected_features]
    X_valid_subset = X_valid[selected_features]
    X_test_subset = X_test[selected_features]
    model_name = 'model_11_features'
    report = 'report_11_features.txt'
    logging.info(f"\nTraining ensemble model with 11 features...")
    logging.info(f"Memory before training: {get_memory_usage()}")
    feature_train_start = time.time()
    try:
        acc_train, acc_valid, acc_test, f1_train, f1_valid, f1_test = train_and_evaluate_ensemble(
            X_train_subset, y_train, X_valid_subset, y_validate, X_test_subset, y_test, 
            pathC, model_name, report, ENSEMBLE_SEEDS
        )
        feature_time = convert(time.time() - feature_train_start)
        logging.info(f"Training with 11 features (ensemble) completed in {feature_time}")
        logging.info(f"Memory after training: {get_memory_usage()}")
        results = [{
            'num_features': 11,
            'acc_train': acc_train,
            'acc_valid': acc_valid,
            'acc_test': acc_test,
            'f1_train': f1_train,
            'f1_valid': f1_valid,
            'f1_test': f1_test
        }]
    except Exception as e:
        logging.error(f"Error training ensemble model with 11 features: {str(e)}")
        raise
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(pathC, 'metrics_11_features.csv'), index=False)
    print(f"Results saved to {os.path.join(pathC, 'metrics_11_features.csv')}")
    end = time.time()
    time_taken = convert(end-start)
    logging.info(f"Total processing time: {time_taken}")
    logging.info(f"Final memory usage: {get_memory_usage()}")
    logging.info(f"Log file saved to: {log_file}")
    print(f"\nScript completed successfully. Check the log file at: {log_file}")
