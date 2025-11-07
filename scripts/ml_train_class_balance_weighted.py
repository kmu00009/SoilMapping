# -*- coding: utf-8 -*-
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
import psutil
import logging
from datetime import datetime
from pathlib import Path

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
        tree_method='hist',
        use_label_encoder=False
    )
    
    # Define hyperparameter grid (slightly increase model complexity, reduce regularization)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'min_child_weight': [7, 10, 15],
        'gamma': [1.0, 2.0, 5.0],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # expanded range
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # expanded range
        'reg_lambda': [10, 50, 100, 200],
        'reg_alpha': [5, 10, 20]
    }
    
    # Early stopping fit params
    fit_params = {
        "eval_set": [(X_valid, y_validate)],
        "early_stopping_rounds": 10,
        "sample_weight": sample_weights
    }
    
    # Perform hyperparameter tuning (increase n_iter for more thorough search)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,  # Increased for more thorough search
        cv=10,      # Increased folds for more robust validation
        scoring='f1_weighted',
        n_jobs=-1,  # Parallel processing
        verbose=1,
        random_state=42,
        error_score='raise',
        refit=True
    )
    random_search.fit(X_train, y_train, **fit_params)
    
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
        early_stopping_rounds=10,
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
        tree_method='hist',
        use_label_encoder=False
    )
    sk_model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_valid, y_validate)], early_stopping_rounds=10, verbose=False)
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
    cm_train = confusion_matrix(y_train, y_pred_train)
    cr_train = classification_report(y_train, y_pred_train)
    print(f"\nAccuracy on training data ({num_features} features): {accuracy_train * 100:.2f}%")
    print(f"F1-Score (weighted) on training data: {f1_train * 100:.2f}%")
    print("Confusion Matrix (Train):")
    print(cm_train)
    print("Classification Report (Train):")
    print(cr_train)
    
    # Evaluate on validation data
    y_pred_valid = model.predict(dvalid).astype(int)
    accuracy_valid = round(accuracy_score(y_validate, y_pred_valid), 2)
    f1_valid = round(f1_score(y_validate, y_pred_valid, average='weighted'), 2)
    cm_valid = confusion_matrix(y_validate, y_pred_valid)
    cr_valid = classification_report(y_validate, y_pred_valid)
    print(f"Accuracy on validation data ({num_features} features): {accuracy_valid * 100:.2f}%")
    print(f"F1-Score (weighted) on validation data: {f1_valid * 100:.2f}%")
    print("Confusion Matrix (Validation):")
    print(cm_valid)
    print("Classification Report (Validation):")
    print(cr_valid)
    
    # Evaluate on test data
    y_pred_test = model.predict(dtest).astype(int)
    accuracy_test = round(accuracy_score(y_test, y_pred_test), 2)
    f1_test = round(f1_score(y_test, y_pred_test, average='weighted'), 2)
    cm_test = confusion_matrix(y_test, y_pred_test)
    cr_test = classification_report(y_test, y_pred_test)
    print(f"Accuracy on test data ({num_features} features): {accuracy_test * 100:.2f}%")
    print(f"F1-Score (weighted) on test data: {f1_test * 100:.2f}%")
    print("Confusion Matrix (Test):")
    print(cm_test)
    print("Classification Report (Test):")
    print(cr_test)
    
    # Log overfitting diagnosis
    print(f"Overfitting gap (train - test accuracy): {accuracy_train - accuracy_test:.2f}")
    print(f"Overfitting gap (train - test F1): {f1_train - f1_test:.2f}")
    
    # Write report with confusion matrices and classification reports
    report_content = f"""
Results for {num_features} features:
Best parameters: {best_params}
Permutation Feature Importances:
{feature_importance_df.to_string()}
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
Overfitting gap (train - test accuracy): {accuracy_train - accuracy_test:.2f}
Overfitting gap (train - test F1): {f1_train - f1_test:.2f}
"""
    # First read existing content if file exists
    try:
        if os.path.exists(os.path.join(pathC, report)):
            with open(os.path.join(pathC, report), 'r') as f:
                existing_content = f.read()
        else:
            existing_content = ""
    except:
        existing_content = ""
    
    # Write complete content (existing + new)
    try:
        with open(os.path.join(pathC, report), 'w') as f:
            f.write(existing_content + report_content)
        logging.info(f"Report updated: {os.path.join(pathC, report)}")
    except Exception as e:
        logging.error(f"Could not write to {report}. Error: {str(e)}")
        # Fallback: write to a different location
        fallback_path = os.path.join('/dbfs/FileStore/SoilMapping/reports/', report)
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path, 'w') as f:
            f.write(existing_content + report_content)
        logging.info(f"Report written to fallback location: {fallback_path}")
    
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
    pathC = os.environ.get('CLASSIFICATION_DIR', '/dbfs/mnt/lab/unrestricted/KritiM/classification/')
    training_file = os.environ.get('TRAINING_FILE', os.path.join(pathC, 'trainingSample.csv'))
    os.makedirs(pathC, exist_ok=True)

    # Delete all files in pathC except the training file
    # Convert to dbfs:/ path for dbutils.fs
    dbfs_pathC = pathC.replace('/dbfs', 'dbfs:') if pathC.startswith('/dbfs') else pathC
    keep_file = Path(training_file).name
    try:
        files = dbutils.fs.ls(dbfs_pathC)
        for f in files:
            if f.name != keep_file:
                dbutils.fs.rm(f.path, True)
    except Exception as e:
        print(f"Warning: Could not clean directory {dbfs_pathC}. Error: {e}")

    # Setup logging
    log_file = setup_logging(pathC)
    logging.info(f"Starting training script - Output directory: {pathC}")
    import sklearn
    logging.info(f"Python packages: xgboost {xgb.__version__}, sklearn {sklearn.__version__}, pandas {pd.__version__}")
    logging.info(f"Initial memory usage: {get_memory_usage()}")
    
    # Load data
    logging.info('Loading the labelled data...')
    try:
        df = pd.read_csv(training_file)
        logging.info(f"Successfully loaded data from {training_file}")
        logging.info(f"Initial dataframe shape: {df.shape}")
        logging.info(f"Memory usage after load: {get_memory_usage()}")
        
        # df = df.drop(columns=['Land_cover'], errors='ignore')
        df = df.drop_duplicates()
        df = df.dropna()
        logging.info(f"Shape after cleaning: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading/cleaning data: {str(e)}")
        raise
    
    # Prepare target and features
    df['target'] = df['target'].astype(int) - 1  # Shift to zero-based
    print("Class distribution:")
    print(df['target'].value_counts(normalize=True))
    
    print('assign categorical and numerical columns...')
    categorical_cols = ['Land_cover', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                        'Aggregate_texture', 'Aquifers', 'bedrock']
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

    # --- SMOTENC augmentation for sparse/low-F1 classes ---
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError:
        import sys
        !{sys.executable} -m pip install imbalanced-learn
        from imblearn.over_sampling import SMOTENC

    # Identify sparse classes (<50% of majority class)
    class_counts = y_train.value_counts()
    majority_count = class_counts.max()
    sparse_classes = class_counts[class_counts < 0.5 * majority_count].index.tolist()

    # (Optional) Identify classes with F1 < 0.5 (if available from previous runs)
    # For now, only use sparse_classes as F1s are not available at this stage
    augment_classes = set(sparse_classes)
    if len(augment_classes) > 0:
        print(f"Applying SMOTENC to augment classes: {augment_classes}")
        # Get categorical feature indices for SMOTENC
        cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_cols]
        # For multi-class, sampling_strategy must be a dict: {class_label: n_samples}
        target_n = int(np.ceil(0.3 * majority_count))
        # Only include classes where target_n > current count (SMOTE only oversamples)
        sampling_strategy = {cls: target_n for cls in augment_classes if target_n > class_counts[cls]}
        if len(sampling_strategy) == 0:
            print("No classes require oversampling with SMOTENC (target_n <= current count for all). Skipping SMOTENC.")
        else:
            smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42, n_jobs=-1, sampling_strategy=sampling_strategy)
            # Only oversample the selected classes
            # Create a mask for samples to keep (majority and sparse)
            mask = y_train.isin(sampling_strategy.keys()) | y_train.isin([class_counts.idxmax()])
            X_train_aug, y_train_aug = smote_nc.fit_resample(X_train[mask], y_train[mask])
            # Concatenate with the rest of the data (not oversampled)
            X_train_rest = X_train[~mask]
            y_train_rest = y_train[~mask]
            X_train_final = pd.concat([pd.DataFrame(X_train_aug, columns=X_train.columns), X_train_rest], ignore_index=True)
            y_train_final = pd.concat([pd.Series(y_train_aug), y_train_rest], ignore_index=True)
            X_train = X_train_final
            y_train = y_train_final
            print(f"Training set shape after SMOTENC: {X_train.shape}")
    else:
        print("No sparse classes found for SMOTENC augmentation.")

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
    feature_counts = range(13, 14, 1)
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
        logging.info(f"\nTraining model with {k} features...")
        logging.info(f"Memory before training: {get_memory_usage()}")
        feature_train_start = time.time()
        
        try:
            acc_train, acc_valid, acc_test, f1_train, f1_valid, f1_test = train_and_evaluate(
                X_train_subset, y_train, X_valid_subset, y_validate, X_test_subset, y_test, 
                pathC, model_name, report, k
            )
            feature_time = convert(time.time() - feature_train_start)
            logging.info(f"Training with {k} features completed in {feature_time}")
            logging.info(f"Memory after training: {get_memory_usage()}")
            
            # Store results in a dictionary
            result_entry = {}
            result_entry['num_features'] = k
            result_entry['acc_train'] = acc_train
            result_entry['acc_valid'] = acc_valid
            result_entry['acc_test'] = acc_test
            result_entry['f1_train'] = f1_train
            result_entry['f1_valid'] = f1_valid
            result_entry['f1_test'] = f1_test
            results.append(result_entry)
        except Exception as e:
            logging.error(f"Error training model with {k} features: {str(e)}")
            raise
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(pathC, 'metrics_vs_features.csv'), index=False)
    print(f"Results saved to {os.path.join(pathC, 'metrics_vs_features.csv')}")
    
    # Plot metrics
    # plot_metrics(
    #     results_df['num_features'],
    #     results_df['acc_train'],
    #     results_df['acc_valid'],
    #     results_df['acc_test'],
    #     results_df['f1_train'],
    #     results_df['f1_valid'],
    #     results_df['f1_test'],
    #     pathC
    # )
    
    # Suggestions for further improvement (not implemented in code):
    # - Double-check that none of the top features are data leaks (e.g., derived from the target).
    # - If overfitting persists, try advanced feature engineering (interactions, polynomials, domain-specific transforms).
    # - Consider ensemble averaging of several top models if feasible.
    # - Plot learning curves to visually inspect overfitting.
    
    end = time.time()
    time_taken = convert(end-start)
    logging.info(f"Total processing time: {time_taken}")
    logging.info(f"Final memory usage: {get_memory_usage()}")
    logging.info(f"Log file saved to: {log_file}")
    print(f"\nScript completed successfully. Check the log file at: {log_file}")