# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:05:31 2025

@author: kriti.mukherjee
"""

import time
import pandas as pd
import multiprocessing
import glob
import os
import shutil
import joblib
import numpy as np
import rasterio as rio
import argparse
from pathlib import Path
from rasterio.merge import merge
import re  # For extract_index
import lightgbm as lgb  # For LightGBM model
import logging  # For logging
import uuid  # For generating unique temporary file names

# Set up logging to output to console at INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def clear_directory(directory):
    directory = Path(directory)
    if not directory.exists():
        logger.info(f"Directory {directory} does not exist.")
        return
    if not directory.is_dir():
        logger.info(f"{directory} is not a directory.")
        return
    for item in directory.iterdir():
        try:
            if item.is_file():
                item.unlink()
                logger.info(f"Removed file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                logger.info(f"Removed directory: {item}")
        except Exception as e:
            logger.error(f"Error removing {item}: {e}")

def get_model_feature_names(model_path):
    try:
        model = joblib.load(model_path)
        # For LightGBM scikit-learn API
        if hasattr(model, 'feature_name_'):
            return list(model.feature_name_)
        # For LightGBM native Booster
        elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
            return list(model.booster_.feature_name())
        # For scikit-learn
        elif hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        else:
            logger.info("Model does not have feature_names attribute.")
            return None
    except Exception as e:
        logger.error(f"Error loading model or extracting feature names: {e}")
        return None

def extract_index(filename):
    match = re.search(r'class_(\d+)\.csv', filename)
    return int(match.group(1)) if match else -1

def split_dataframe(df, chunk_size):
    return [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def align_features_for_scaler(df, scaler_feature_names):
    # Add missing columns as NaN
    for col in scaler_feature_names:
        if col not in df.columns:
            df[col] = np.nan
    # Remove extra columns
    df = df[scaler_feature_names]
    return df

def predictClass(infile, outpath, i, classifier, scaler, model_feature_names):
    try:
        df = pd.read_csv(infile)
        msg = f"Processing chunk {i}"
        print(msg)
        logger.info(msg)
        nan_mask = df.isna().any(axis=1)
        df_filled = df.fillna(0)
        # Debug: print model features and DataFrame columns
        print(f"DEBUG: Model expects features: {model_feature_names}")
        print(f"DEBUG: DataFrame columns: {df_filled.columns.tolist()}")
        logger.info(f"Model expects features: {model_feature_names}")
        logger.info(f"Prediction DataFrame columns: {df_filled.columns.tolist()}")
        # Print model n_features_in_ if available
        if hasattr(classifier, 'n_features_in_'):
            print(f"DEBUG: Model n_features_in_: {classifier.n_features_in_}")
            logger.info(f"Model n_features_in_: {classifier.n_features_in_}")
        # Fail-fast: Only proceed if columns match exactly (names and order)
        if list(df_filled.columns) != list(model_feature_names):
            logger.error(f"Column mismatch: Model expects {model_feature_names}, got {df_filled.columns.tolist()}")
            print(f"ERROR: Column mismatch: Model expects {model_feature_names}, got {df_filled.columns.tolist()}")
            return  # Skip this chunk
        if hasattr(classifier, 'n_features_in_') and len(model_feature_names) != classifier.n_features_in_:
            logger.error(f"Model expects {classifier.n_features_in_} features, but model_feature_names has {len(model_feature_names)} features.")
            print(f"ERROR: Model expects {classifier.n_features_in_} features, but model_feature_names has {len(model_feature_names)} features.")
            return
        df_pred = df_filled[model_feature_names]
        logger.info(f"Columns in df_pred: {df_pred.columns.tolist()}")
        class_values = classifier.predict(df_pred)
        if hasattr(classifier, 'predict_proba'):
            prob_values = classifier.predict_proba(df_pred)
            confidence_values = np.max(prob_values, axis=1)
        else:
            prob_values = classifier.predict(df_pred)
            if prob_values.ndim == 1:
                confidence_values = prob_values
                class_values = (prob_values > 0.5).astype(int)
            else:
                confidence_values = np.max(prob_values, axis=1)
        msg = f"Chunk {i} confidence range: {confidence_values.min():.3f} to {confidence_values.max():.3f}"
        print(msg)
        logger.info(msg)
        # Add 1 to class if needed (for 1-based class)
        class_values = class_values + 1
        # Convert class_values to float to allow np.nan assignment
        class_values = class_values.astype(float)
        # Set nan where original data was nan
        class_values[nan_mask] = np.nan
        confidence_values[nan_mask] = np.nan
        output_file = os.path.join(outpath, f'class_{i}.csv')
        df_filled['Class'] = class_values
        df_filled['Confidence'] = confidence_values
        df_filled.loc[nan_mask, ['Class', 'Confidence']] = np.nan
        df_filled.to_csv(output_file, index=False)
        msg = f"Saved predictions to {output_file}"
        print(msg)
        logger.info(msg)
    except Exception as e:
        msg = f"An error occurred: {e}"
        print(msg)
        logger.error(msg)
        raise

def mosaic_rasters(input_files, output_file, nodata_value=-9999):
    import shutil
    import uuid
    src_files_to_mosaic = []
    for fp in input_files:
        src = rio.open(fp)
        src_files_to_mosaic.append(src)
    if not src_files_to_mosaic:
        logger.warning(f"No input files found for {output_file}, skipping.")
        return
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata_value)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "nodata": nodata_value,
        "count": 1
    })
    # Write to /tmp first, then copy to DBFS
    tmp_mosaic_path = f"/tmp/mosaic_{uuid.uuid4().hex}.tif"
    with rio.open(tmp_mosaic_path, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)
    shutil.copy(tmp_mosaic_path, output_file)
    os.remove(tmp_mosaic_path)
    for src in src_files_to_mosaic:
        src.close()
    logger.info(f"Mosaic written to {output_file}")

def main():
    start = time.time()
    pathC = Path(os.environ.get('CLASSIFICATION_DIR', '/dbfs/mnt/lab/unrestricted/KritiM/classification/'))
    training_file = Path(os.environ.get('TRAINING_FILE', str(pathC / 'trainingSample.csv')))
    model_path = Path(os.environ.get('MODEL_PATH', str(pathC / 'model_9_features.joblib')))
    scaler_path = Path(os.environ.get('SCALER_PATH', str(pathC / 'scaler.joblib')))
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    scaler_feature_names = scaler.feature_names_in_.tolist()
    model_feature_names = get_model_feature_names(model_path)
    if model_feature_names is None:
        logger.info("Falling back to scaler feature names for model.")
        model_feature_names = scaler_feature_names
    categorical_cols = ['Landcover_LE', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                        'Aggregate_texture', 'Aquifers', 'bedrock_raster_50m', 'ALC_old']
    dftrain = pd.read_csv(training_file)
    mode_values = {col: dftrain[col].mode()[0] for col in categorical_cols if col in dftrain.columns}
    mean_values = dftrain.select_dtypes(include='number').mean()
    traincols = dftrain.columns.tolist()
    pathtogrids = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID')
    if not pathtogrids.exists() or not pathtogrids.is_dir():
        logger.error(f"Directory {pathtogrids} does not exist. Please check the path or create the directory and try again.")
        print(f"ERROR: Directory {pathtogrids} does not exist. Please check the path or create the directory and try again.")
        return
    subdirectories = [subdir for subdir in pathtogrids.iterdir() if subdir.is_dir()]
    subdirectories.sort()
    outdir = Path('/dbfs/mnt/lab/unrestricted/KritiM/') / 'Table'
    outdir.mkdir(parents=True, exist_ok=True)
    predict_dir = Path('/dbfs/mnt/lab/unrestricted/KritiM/Predict')
    predict_dir.mkdir(parents=True, exist_ok=True)
    pathraster = pathtogrids / 'AAR'
    files = list(pathraster.glob('*.tif'))

    # --- NEW: Collect all possible variable names from all grids ---
    all_possible_vars = set()
    for file in files:
        grid = file.name[:5]
        for folder in subdirectories:
            files_grid = [file for file in folder.glob(grid + '*.tif') if file.is_file()]
            for file_grid in files_grid:
                var = file_grid.name[6:-4]
                all_possible_vars.add(var)
    all_possible_vars = sorted(list(all_possible_vars))
    # Add EAST and NORTH as they are always present
    all_possible_vars = ['EAST', 'NORTH'] + [v for v in all_possible_vars if v not in ['EAST', 'NORTH']]
    logger.info(f"All possible variables (columns) from GRID: {all_possible_vars}")

    nodata_value = -9999.0  # Use a finite float for nodata
    class_raster_files = []
    conf_raster_files = []
    for file in files:
        grid = file.name[:5]
        csv_path = outdir / (grid + '.csv')
        class_raster_path = predict_dir / f'{grid}_class.tif'
        conf_raster_path = predict_dir / f'{grid}_conf.tif'
        # Skip processing if all files exist
        if csv_path.exists() and class_raster_path.exists() and conf_raster_path.exists():
            logger.info(f"Skipping grid {grid}: CSV and both rasters already exist.")
            class_raster_files.append(str(class_raster_path))
            conf_raster_files.append(str(conf_raster_path))
            continue
        output_file = predict_dir / f'{grid}_predict_xgb.tif'
        if not output_file.exists():
            df = pd.DataFrame()
            for folder in subdirectories:
                logger.info(f'Working on folder: {folder}')
                files_grid = [file for file in folder.glob(grid + '*.tif') if file.is_file()]
                if not files_grid:
                    logger.info(f"No files found for grid {grid} in {folder}")
                    continue
                for file_grid in files_grid:
                    grid_name = file_grid.name[:5]
                    var = file_grid.name[6:-4]
                    logger.info(f"Processing file: {file_grid.name}, Grid: {grid_name}, Variable: {var}")
                    with rio.open(str(file_grid), 'r') as src:
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
            logger.info(f'Created dataframe for {grid}...')
            logger.info(f"Dataframe columns: {df.columns.tolist()}")
            # --- Ensure all columns are present, fill missing with NaN, consistent order ---
            df_to_save = df.reindex(columns=all_possible_vars)
            logger.info(f"Saving all possible columns for grid {grid}: {df_to_save.columns.tolist()}")
            csv_path = outdir / (grid + '.csv')
            df_to_save.to_csv(csv_path, index=False)

            # --- NEW: Predict class and confidence, save as raster ---
            # Only use model features for prediction
            df_pred = df_to_save[model_feature_names].copy()
            nan_mask = df_pred.isna().any(axis=1)
            df_pred_filled = df_pred.fillna(0)
            # --- FIX: Ensure categorical columns have same dtype and categories as training ---
            for col in categorical_cols:
                if col in df_pred_filled.columns and col in dftrain.columns:
                    train_cats = pd.Categorical(dftrain[col]).categories
                    df_pred_filled[col] = pd.Categorical(df_pred_filled[col], categories=train_cats)
            # Predict
            class_values = best_model.predict(df_pred_filled)
            if hasattr(best_model, 'predict_proba'):
                prob_values = best_model.predict_proba(df_pred_filled)
                confidence_values = np.max(prob_values, axis=1)
            else:
                prob_values = best_model.predict(df_pred_filled)
                if prob_values.ndim == 1:
                    confidence_values = prob_values
                    class_values = (prob_values > 0.5).astype(int)
                else:
                    confidence_values = np.max(prob_values, axis=1)
            # Add 1 to class if needed (for 1-based class)
            class_values = class_values + 1
            # Convert class_values to float to allow nodata assignment
            class_values = class_values.astype(float)
            # Set nodata_value where original data was nan
            class_values[nan_mask] = nodata_value
            confidence_values[nan_mask] = nodata_value
            # Reshape to raster
            # Use the shape and nodata mask from the reference raster (AAR) in the grid
            first_raster = None
            ref_nodata = None
            ref_mask = None
            for folder in subdirectories:
                files_grid = [file for file in folder.glob(grid + '*.tif') if file.is_file()]
                if files_grid:
                    # Prefer the AAR raster as reference if available
                    aar_raster = [f for f in files_grid if f.name.startswith(f'{grid}_AAR')]
                    if aar_raster:
                        first_raster = aar_raster[0]
                    else:
                        first_raster = files_grid[0]
                    break
            if first_raster is not None:
                with rio.open(str(first_raster), 'r') as src:
                    height, width = src.height, src.width
                    transform = src.transform
                    crs = src.crs
                    ref_nodata = src.nodata
                    ref_data = src.read(1)
                    # Create mask: True where input is nodata
                    if ref_nodata is not None:
                        ref_mask = (ref_data == ref_nodata)
                    else:
                        ref_mask = np.zeros((height, width), dtype=bool)
                    class_raster = np.ascontiguousarray(class_values.reshape((height, width)).astype(np.float32))
                    conf_raster = np.ascontiguousarray(confidence_values.reshape((height, width)).astype(np.float32))
                    # Mask output: set to nodata_value where input is nodata
                    class_raster[ref_mask] = nodata_value
                    conf_raster[ref_mask] = nodata_value
                    # Replace any remaining np.nan with nodata_value (shouldn't be needed, but safe)
                    class_raster[np.isnan(class_raster)] = nodata_value
                    conf_raster[np.isnan(conf_raster)] = nodata_value
                    class_raster_path = predict_dir / f'{grid}_class.tif'
                    conf_raster_path = predict_dir / f'{grid}_conf.tif'
                    meta = src.meta.copy()
                    meta.update({
                        'count': 1,
                        'dtype': 'float32',
                        'nodata': nodata_value,
                        'driver': 'GTiff',
                        'height': height,
                        'width': width,
                        'transform': transform,
                        'crs': crs,
                        'tiled': True,
                        'blockxsize': 256,
                        'blockysize': 256,
                        'BIGTIFF': 'YES',
                    })
                    # Write to /tmp first, then copy to DBFS
                    import shutil
                    tmp_class_path = f'/tmp/{grid}_class.tif'
                    tmp_conf_path = f'/tmp/{grid}_conf.tif'
                    with rio.open(tmp_class_path, 'w', **meta) as dst:
                        dst.write(class_raster, 1)
                    with rio.open(tmp_conf_path, 'w', **meta) as dst:
                        dst.write(conf_raster, 1)
                    # Copy to DBFS
                    shutil.copy(tmp_class_path, class_raster_path)
                    shutil.copy(tmp_conf_path, conf_raster_path)
                    # Clean up temp files
                    os.remove(tmp_class_path)
                    os.remove(tmp_conf_path)
                    class_raster_files.append(str(class_raster_path))
                    conf_raster_files.append(str(conf_raster_path))
                    logger.info(f"Saved class raster: {class_raster_path}")
                    logger.info(f"Saved confidence raster: {conf_raster_path}")
            else:
                logger.warning(f"No raster found for grid {grid} to get shape/transform.")

    # --- Mosaic all class rasters and confidence rasters ---
    if class_raster_files:
        mosaic_class_path = predict_dir / 'mosaic_class.tif'
        mosaic_rasters(class_raster_files, str(mosaic_class_path), nodata_value=nodata_value)
        logger.info(f"Mosaic of class rasters saved to {mosaic_class_path}")
    if conf_raster_files:
        mosaic_conf_path = predict_dir / 'mosaic_conf.tif'
        mosaic_rasters(conf_raster_files, str(mosaic_conf_path), nodata_value=nodata_value)

# Run main if this script is executed (not imported)
if __name__ == "__main__":
    main()