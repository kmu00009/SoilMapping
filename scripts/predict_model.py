# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:25:31 2025

@author: kriti.mukherjee
"""
# %pip install rasterio
import time
import pandas as pd
import glob
import os
import shutil
import joblib
import numpy as np
import rasterio as rio
from rasterio.merge import merge as rio_merge
from pathlib import Path
import xgboost as xgb
import re
import tempfile
import gc
import psutil
from scipy import stats

NODATA_VALUE = -9999  # Use a single constant for nodata throughout

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def split_dataframe(df, chunk_size):
    """Split a DataFrame into smaller DataFrames of a specified chunk size."""
    num_chunks = len(df) // chunk_size + int(len(df) % chunk_size != 0)
    return [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

def predictClass(infile, outpath, i, classifier, scaler, feature_names):
    import os

    try:
        df = pd.read_csv(infile)
        print(f"Processing chunk {i}")

        nan_mask = df.isna().any(axis=1)
        df_filled = df.fillna(0)

        df_pred = df_filled[feature_names]        

        # --- Predict directly ---
        class_values = classifier.predict(df_pred)
        prob_values = classifier.predict_proba(df_pred)

        confidence_values = np.max(prob_values, axis=1)
        print(
            f"Chunk {i} confidence range: "
            f"{confidence_values.min():.3f} to {confidence_values.max():.3f}"
        )

        df_filled['Class'] = class_values + 1
        df_filled['Confidence'] = confidence_values
        df_filled.loc[nan_mask, ['Class', 'Confidence']] = np.nan

        output_file = os.path.join(outpath, f'class_{i}.csv')
        df_filled.to_csv(output_file, index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def extract_index(filename):
    match = re.search(r'class_(\d+)\.csv', filename)
    return int(match.group(1)) if match else -1

def clear_directory(directory):
    """
    Remove all files and subdirectories within the specified directory.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist.")
        return
    if not directory.is_dir():
        print(f"{directory} is not a directory.")
        return
    for item in directory.iterdir():
        try:
            if item.is_file():
                item.unlink()
                print(f"Removed file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
        except Exception as e:
            print(f"Error removing {item}: {e}")

def get_model_feature_names(model_path):
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'feature_names'):
            return model.feature_names
        elif hasattr(model, 'get_booster'):
            return model.get_booster().feature_names
        else:
            print("Model does not have feature_names attribute.")
            return None
    except Exception as e:
        print(f"Error loading model or extracting feature names: {e}")
        return None

def load_ensemble_models(model_dir, pattern='model_11_features_seed*.joblib'):
    model_paths = sorted(Path(model_dir).glob(pattern))
    models = [joblib.load(str(p)) for p in model_paths]
    if not models:
        raise RuntimeError(f"No ensemble models found in {model_dir} with pattern {pattern}")
    return models


def predict(grid, seed):
    # Load the pre-trained model and scaler
    try:
        model_path = Path(f'/dbfs/mnt/lab/unrestricted/KritiM/classification/model_11_features_seed{seed}.joblib')        
        scaler_path = Path('/dbfs/mnt/lab/unrestricted/KritiM/classification/scaler.joblib')
        best_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        scaler_feature_names = scaler.feature_names_in_.tolist()
        print(f"Scaler feature names (full numerical): {scaler_feature_names}")
        model_feature_names = get_model_feature_names(model_path)
        if model_feature_names is None:
            print("Falling back to scaler feature names for model.")
            model_feature_names = scaler_feature_names
        print(f"Model feature names (subset): {model_feature_names}")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        raise
    
    # Define categorical columns
    categorical_cols = ['Landcover_LE', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                    'Aggregate_texture', 'Aquifers', 'bedrock_raster_50m', 'ALC_old']
    
    # Prepare data for classification
    train_path = Path('/dbfs/mnt/lab/unrestricted/KritiM/classification/trainingSample.csv')
    dftrain = pd.read_csv(train_path)    
    mode_values = {col: dftrain[col].mode()[0] for col in categorical_cols if col in dftrain.columns}
    mean_values = dftrain.select_dtypes(include='number').mean()
    traincols = dftrain.columns.tolist()
    pathtogrids = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID/')
    subdirectories = [subdir for subdir in pathtogrids.iterdir() if subdir.is_dir()]
    subdirectories.sort()
    
    # Write dataframe to CSV
    outpath = Path('/dbfs/mnt/lab/unrestricted/KritiM')
    outdir = outpath / 'Table'
    if not os.path.exists(outdir / (grid + '.csv')):
        df = pd.DataFrame()
        for folder in subdirectories:
            print(f'Working on folder: {folder}')
            files = [file for file in folder.glob(grid + '*.tif') if file.is_file()]
            if not files:
                print(f"No files found for grid {grid} in {folder}")
                continue
            for file in files:
                grid_name = file.name[:5]
                var = file.name[6:-4]
                print(f"Processing file: {file.name}, Grid: {grid_name}, Variable: {var}")
                with rio.open(file, 'r') as src:
                    data = src.read(1).ravel()
                    if 'EAST' not in df.columns:
                        rows, cols = np.meshgrid(
                            np.arange(src.height),
                            np.arange(src.width),
                            indexing="ij"
                        )
                        xs, ys = rio.transform.xy(src.transform, rows, cols)
                        df['EAST'] = np.array(xs).ravel()
                        df['NORTH'] = np.array(ys).ravel()
                    df[var] = data
                    
        print(f'Created dataframe for {grid}...')
        print("Dataframe columns:", df.columns.tolist())
        
        # Check for missing features
        expected_features = scaler_feature_names + [col for col in categorical_cols if col in df.columns]
        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            print(f"Error: Missing features for grid {grid}: {missing_features}")
            return        
        
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory {outdir}: {e}")
            raise
        df.to_csv(outdir / (grid + '.csv'), index=False)
    
    # Read one of the raster grid files for profile information
    pathraster = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID/AAR')
    raster = f"{grid}_AAR.tif"
    try:
        with rio.open(os.path.join(pathraster, raster), 'r') as src:
            profile = src.profile
            profile.update(count=1)
            band = src.read(1)
            nodata_value = src.nodatavals[0] if src.nodatavals else -9999
    except Exception as e:
        print(f"Error reading raster file: {e}")
        raise
    
    # Read the dataframe for prediction
    pathdata = Path('/dbfs/mnt/lab/unrestricted/KritiM/Table')
    df = pd.read_csv(pathdata / f"{grid}.csv")
    df = df.replace(nodata_value, np.nan)
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Check for valid rows
    print(f"Total rows in df: {len(df)}")
    valid_mask = df.notna().all(axis=1)
    print(f"Valid rows (no NaNs): {valid_mask.sum()}")
    print("NaN counts per column:")
    print(df.isna().sum())
    df_valid = df[valid_mask].copy()
    
    if df_valid.empty:
        print(f"No valid data (all rows contain NaNs) for grid {grid}. Skipping prediction.")
        return
    
    # Downcast numerical columns
    for col in df_valid.select_dtypes(include='number').columns:
        df_valid[col] = pd.to_numeric(df_valid[col], downcast='float')
    print("Columns in df_valid:", df_valid.columns.tolist())
    
    # Scale numerical columns
    num_cols = [col for col in scaler_feature_names if col not in categorical_cols]
    try:
        print('Scaling the prediction dataframe...')
        df_num_scaled = pd.DataFrame(
            scaler.transform(df_valid[num_cols]),
            columns=num_cols, index=df_valid.index
        )
        df_scaled = pd.concat([df_num_scaled, df_valid[categorical_cols]], axis=1)
        print('Scaling completed...')
        
        # Select only expected features
        df_scaled_select = df_scaled[model_feature_names]
        print("Columns in scaled dataset:", df_scaled_select.columns.tolist())
        print('Shape of dataframe:', df_scaled_select.shape)
        
        chunk_size = 100000
        chunks = split_dataframe(df_scaled_select, chunk_size)
        
        tmp = Path(f'/dbfs/mnt/lab/unrestricted/KritiM/Predict_{seed}') / grid
        tmp.mkdir(parents=True, exist_ok=True)
        
        inFiles = list(tmp.glob('data_*.csv'))
        if not inFiles:
            for i, chunk in enumerate(chunks):
                chunk.to_csv(tmp / f'data_{i}.csv', index=False)
            inFiles = list(tmp.glob('data_*.csv'))
        
        for i, file in enumerate(inFiles):
            print(f'Predicting classification for grid {grid}, chunk {i}')
            predictClass(file, tmp, i, best_model, scaler, model_feature_names)
        
        print('Merging the classified data...')
        toMergeC = list(tmp.glob('class_*.csv'))
        if not toMergeC:
            print(f"No class_*.csv files found for grid {grid}. Skipping merge step.")
            return
        
        toMergeCF = sorted(toMergeC, key=lambda x: extract_index(str(x)))
        dfsC = []
        for i in toMergeCF:
            print(f"Reading dataframe {i}")
            df_chunk = pd.read_csv(i)
            if 'Confidence' not in df_chunk.columns:
                print(f"Error: 'Confidence' column missing in {i}")
                return
            subset = df_chunk[['Class', 'Confidence']]
            dfsC.append(subset)
        MergedC = pd.concat(dfsC)
        
        # Create full prediction and confidence arrays with NaNs
        full_class_pred = np.full(shape=len(df), fill_value=np.nan)
        full_confidence_pred = np.full(shape=len(df), fill_value=np.nan)
        full_class_pred[valid_mask] = MergedC['Class'].values
        full_confidence_pred[valid_mask] = MergedC['Confidence'].values
        
        # Validate confidence range
        print(f"Confidence range for grid {grid}: {np.nanmin(full_confidence_pred):.3f} to {np.nanmax(full_confidence_pred):.3f}")
        
        # Reshape the 'Class' and 'Confidence' arrays
        S_class = np.reshape(full_class_pred, (band.shape[0], band.shape[1]))
        S_confidence = np.reshape(full_confidence_pred, (band.shape[0], band.shape[1]))
        
        # Apply the NoData mask
        if nodata_value is not None:
            S_class[band == nodata_value] = nodata_value
            S_confidence[band == nodata_value] = nodata_value
        
        # Save the classified image
        tmp_dir = tempfile.mkdtemp()
        tmp_predict = os.path.join(tmp_dir, f'{grid}_predict_xgb.tif')
        with rio.open(tmp_predict, 'w', **profile) as dst:
            dst.write(S_class.astype(rio.float32), 1)
        
        # Copy to DBFS mount
        predict_path = f'/dbfs/mnt/lab/unrestricted/KritiM/Predict_{seed}/{grid}_predict_xgb.tif'
        shutil.copy2(tmp_predict, predict_path)

        # Clean up
        shutil.rmtree(tmp_dir)


        # Save the confidence image
        tmp_dir = tempfile.mkdtemp()
        tmp_conf = os.path.join(tmp_dir, f'{grid}_confidence_xgb.tif')
        with rio.open(tmp_conf, 'w', **profile) as dst:
            dst.write(S_confidence.astype(rio.float32), 1)

        conf_path = f'/dbfs/mnt/lab/unrestricted/KritiM/Predict_{seed}/{grid}_confidence_xgb.tif'
        shutil.copy2(tmp_conf, conf_path)
        
        # Delete the temporary folder
        shutil.rmtree(tmp_dir)

        # Clean up the grid folder
        shutil.rmtree(tmp)

    except Exception as e:
        print(f'Error predicting for {grid}: {e}')
        return

# Main execution
start = time.time()

path = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID/AAR')
files = list(path.glob('*.tif'))
grids = [file.name[:5] for file in files]
seeds = [7, 42, 99, 123, 2024]

for seed in seeds:
    for grid in grids:
        output_file = Path(f'/dbfs/mnt/lab/unrestricted/KritiM/Predict_{seed}') / f'{grid}_predict_xgb.tif'
        if not output_file.exists():
            predict(grid, seed)

    predict_dir = Path(f'/dbfs/mnt/lab/unrestricted/KritiM/Predict_{seed}')
    mosaic_dir = predict_dir / 'Mosaic'
    mosaic_dir.mkdir(exist_ok=True)

    # List all prediction and confidence rasters
    predict_rasters = sorted(predict_dir.glob('*_predict_xgb.tif'))
    confidence_rasters = sorted(predict_dir.glob('*_confidence_xgb.tif'))

    def mosaic_and_save(raster_files, out_name, method='first'):
        src_files = [rio.open(str(fp)) for fp in raster_files]
        try:
            mosaic, out_trans = rio_merge(src_files, method=method)
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": 1
            })
            # Write to local temp file first
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, str(out_name))  # Ensure out_name is string
            with rio.open(temp_path, "w", **out_meta) as dest:
                dest.write(mosaic[0], 1)
            # Copy to DBFS
            final_path = str(mosaic_dir / str(out_name))
            shutil.copy2(temp_path, final_path)
        finally:
            shutil.rmtree(temp_dir)
            for src in src_files:
                src.close()

    # Mosaic prediction rasters (use 'first' for class prediction)
    mosaic_and_save(
        predict_rasters,
        'soil_predict_xgb_mosaic.tif',
        method='first'
    )

    # Mosaic confidence rasters (use 'max' for confidence)
    mosaic_and_save(
        confidence_rasters,
        'soil_confidence_xgb_mosaic.tif',  # FIX: pass as string, not Path
        method='max'
    )