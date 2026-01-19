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

NODATA_VALUE = -9999  # Use a single constant for nodata throughout

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def split_dataframe(df, chunk_size):
    num_chunks = len(df) // chunk_size + int(len(df) % chunk_size != 0)
    return [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

def extract_index(filename):
    match = re.search(r'class_(\d+)\.csv', filename)
    return int(match.group(1)) if match else -1

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

def predictClass_chunked_ensemble(df, models, scaler, feature_names, chunk_size=100000):
    class_preds = []
    confidence_preds = []
    nan_masks = []
    class_proba_maps = []  # For storing probability maps for each class
    n_classes = 12  # Fixed number of output classes
    for chunk in split_dataframe(df, chunk_size):
        nan_mask = chunk.isna().any(axis=1)
        chunk_filled = chunk.fillna(0)
        chunk_pred = chunk_filled[feature_names]
        # Get probability predictions from all models
        probas = []
        for model in models:
            temp_model_file = 'temp_model.json'
            model.save_model(temp_model_file)
            sk_classifier = xgb.XGBClassifier()
            sk_classifier.load_model(temp_model_file)
            prob_values = sk_classifier.predict_proba(chunk_pred)
            os.remove(temp_model_file)
            probas.append(prob_values)
        avg_proba = np.mean(probas, axis=0)
        class_values = np.argmax(avg_proba, axis=1)
        confidence_values = np.max(avg_proba, axis=1)
        class_preds.append(class_values + 1)  # 1-based
        confidence_preds.append(confidence_values)
        nan_masks.append(nan_mask)
        class_proba_maps.append(avg_proba)
    # Ensure float dtype for NaN assignment
    class_preds = np.concatenate(class_preds).astype(float)
    confidence_preds = np.concatenate(confidence_preds).astype(float)
    nan_mask_full = pd.concat(nan_masks).values if nan_masks else np.array([])
    class_preds[nan_mask_full] = np.nan
    confidence_preds[nan_mask_full] = np.nan
    class_proba_maps = np.concatenate(class_proba_maps, axis=0)
    class_proba_maps[nan_mask_full, :] = np.nan
    return class_preds, confidence_preds, class_proba_maps

def predict_and_write_ensemble(grid_csv, models, scaler, model_feature_names, categorical_cols, output_predict_folder, grid_profile, nodata_value):
    df = pd.read_csv(grid_csv)
    df = df.replace(nodata_value, np.nan)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    valid_mask = df.notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    if df_valid.empty:
        print(f"No valid data (all rows contain NaNs) for {grid_csv}. Skipping prediction.")
        return None
    for col in df_valid.select_dtypes(include='number').columns:
        df_valid[col] = pd.to_numeric(df_valid[col], downcast='float')
    scaler_feature_names = scaler.feature_names_in_.tolist()
    num_cols = [col for col in scaler_feature_names if col not in categorical_cols]
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_valid[num_cols]),
        columns=num_cols, index=df_valid.index
    )
    df_scaled = pd.concat([df_num_scaled, df_valid[categorical_cols]], axis=1)
    df_scaled_select = df_scaled[model_feature_names]
    class_preds, confidence_preds, class_proba_maps = predictClass_chunked_ensemble(df_scaled_select, models, scaler, model_feature_names)
    df.loc[valid_mask, 'Class'] = class_preds
    df.loc[valid_mask, 'Confidence'] = confidence_preds
    # Save the class probability maps for each class
    n_classes = class_proba_maps.shape[1]
    for c in range(n_classes):
        df.loc[valid_mask, f'ClassProb_{c+1}'] = class_proba_maps[:, c]
    df.to_csv(grid_csv, index=False)
    band_shape = (grid_profile['height'], grid_profile['width'])
    class_array = np.full(df.shape[0], np.nan)
    class_array[valid_mask] = class_preds
    class_array = class_array.reshape(band_shape)
    confidence_array = np.full(df.shape[0], np.nan)
    confidence_array[valid_mask] = confidence_preds
    confidence_array = confidence_array.reshape(band_shape)
    class_prob_arrays = []
    for c in range(n_classes):
        arr = np.full(df.shape[0], np.nan)
        arr[valid_mask] = class_proba_maps[:, c]
        arr = arr.reshape(band_shape)
        class_prob_arrays.append(arr)
    if nodata_value is not None:
        with rio.open(grid_profile['raster_path'], 'r') as src:
            band = src.read(1)
        class_array[band == nodata_value] = nodata_value
        confidence_array[band == nodata_value] = nodata_value
        for arr in class_prob_arrays:
            arr[band == nodata_value] = nodata_value
    # Write to local temp directory first to avoid DBFS FUSE issues
    temp_dir = tempfile.mkdtemp()
    temp_raster = os.path.join(temp_dir, f"{grid_profile['grid']}_predict_xgb.tif")
    temp_conf_raster = os.path.join(temp_dir, f"{grid_profile['grid']}_confidence_xgb.tif")
    temp_prob_rasters = [os.path.join(temp_dir, f"{grid_profile['grid']}_classprob_{c+1}_xgb.tif") for c in range(n_classes)]
    profile = grid_profile['profile']
    with rio.open(temp_raster, 'w', **profile) as dst:
        dst.write(class_array.astype(np.float32), 1)
    with rio.open(temp_conf_raster, 'w', **profile) as dst:
        dst.write(confidence_array.astype(np.float32), 1)
    for c, arr in enumerate(class_prob_arrays):
        with rio.open(temp_prob_rasters[c], 'w', **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
    # Copy to final destination in grid_folder
    out_raster = os.path.join(output_predict_folder, f"{grid_profile['grid']}_predict_xgb.tif")
    out_conf_raster = os.path.join(output_predict_folder, f"{grid_profile['grid']}_confidence_xgb.tif")
    out_prob_rasters = [os.path.join(output_predict_folder, f"{grid_profile['grid']}_classprob_{c+1}_xgb.tif") for c in range(n_classes)]
    shutil.copy2(temp_raster, out_raster)
    shutil.copy2(temp_conf_raster, out_conf_raster)
    for c, prob_raster in enumerate(temp_prob_rasters):
        shutil.copy2(prob_raster, out_prob_rasters[c])
    shutil.rmtree(temp_dir)
    print(f"Wrote predicted raster: {out_raster}")
    print(f"Wrote confidence raster: {out_conf_raster}")
    for c, out_prob in enumerate(out_prob_rasters):
        print(f"Wrote class probability raster for class {c+1}: {out_prob}")
    return out_raster

# Main execution
start = time.time()
print('Checking Table folder for existing CSVs for each grid...')

table_folder = Path('/dbfs/mnt/lab/unrestricted/KritiM/Table')
table_folder.mkdir(parents=True, exist_ok=True)
pathtogrids = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID')
tifrefdir = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID/AAR')
# Define subdirectories as all subfolders in the GRID/AAR directory
subdirectories = [f for f in pathtogrids.iterdir() if f.is_dir()]
subdirectories.sort()

# --- New: Write one CSV per grid (e.g., NT_NE, NT_SE, etc.) in Table folder as specified ---
# 1. Find all unique grid names from all .tif files in all subdirectories
all_tif_files = []
for subdir in subdirectories:
    all_tif_files.extend(list(subdir.glob('*.tif')))
grid_names = set()
for tif_file in all_tif_files:
    parts = tif_file.stem.split('_')
    if len(parts) >= 2:
        grid_names.add('_'.join(parts[:2]))

# 2. For each grid, build the DataFrame and write the CSV
for grid_name in sorted(grid_names):
    out_csv = table_folder / f'{grid_name}.csv'
    if out_csv.exists():
        print(f"CSV for grid {grid_name} already exists: {out_csv}, skipping generation.")
        continue
    columns = {}
    coords_written = False
    for subdir in subdirectories:
        tif_files = list(subdir.glob(f'{grid_name}_*.tif'))
        if not tif_files:
            continue
        tif_file = tif_files[0]
        # Extract column name
        suffix = tif_file.stem[len(grid_name)+1:]
        colname = suffix
        with rio.open(tif_file, 'r') as src:
            data = src.read(1).ravel()
            columns[colname] = data
            if not coords_written:
                height, width = src.height, src.width
                rows, cols = np.meshgrid(
                    np.arange(height),
                    np.arange(width),
                    indexing="ij"
                )
                xs, ys = rio.transform.xy(src.transform, rows, cols)
                columns['EAST'] = np.array(xs).ravel()
                columns['NORTH'] = np.array(ys).ravel()
                coords_written = True
    if columns:
        df = pd.DataFrame(columns)
        # Reorder columns: EAST, NORTH, then all raster columns
        col_order = ['EAST', 'NORTH'] + [c for c in columns if c not in ['EAST', 'NORTH']]
        df = df[col_order]
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} with columns: {df.columns.tolist()}")
    else:
        print(f"No data found for grid {grid_name} in any subdirectory.")

# --- CSV column check against raster files for each grid across all subdirectories ---
print("\nChecking that CSV columns match raster files for each grid across all subdirectories...")
csv_files = [f for f in table_folder.glob('*.csv') if f.stem != 'AAR']
for csv_path in csv_files:
    grid = csv_path.stem
    # Collect all raster files for this grid from all subdirectories
    raster_files = []
    for folder in subdirectories:
        raster_files.extend(list(folder.glob(f'{grid}_*.tif')))
    raster_suffixes = set(f.stem[len(grid)+1:] for f in raster_files)
    # Read CSV columns
    df = pd.read_csv(csv_path, nrows=1)  # Only need header
    csv_columns = set(df.columns) - {'EAST', 'NORTH', 'TILE'}
    # Compare
    missing_in_csv = raster_suffixes - csv_columns
    extra_in_csv = csv_columns - raster_suffixes
    if not missing_in_csv and not extra_in_csv:
        print(f"Grid {grid}: All raster files are present as columns in the CSV.")
    else:
        if missing_in_csv:
            print(f"Grid {grid}: Raster files missing as columns in CSV: {sorted(missing_in_csv)}")
        if extra_in_csv:
            print(f"Grid {grid}: CSV columns not found as raster files: {sorted(extra_in_csv)}")

print('Loading ensemble models and scaler...')
pathC = Path(os.environ.get('CLASSIFICATION_DIR', '/dbfs/mnt/lab/unrestricted/KritiM/classification/'))
scaler_path = Path(os.environ.get('SCALER_PATH', str(pathC / 'scaler.joblib')))
scaler = joblib.load(scaler_path)
ensemble_models = load_ensemble_models(pathC, pattern='model_11_features_seed*.joblib')
model_feature_names = get_model_feature_names(list(Path(pathC).glob('model_11_features_seed*.joblib'))[0])
if model_feature_names is None:
    model_feature_names = scaler.feature_names_in_.tolist()

categorical_cols = ['Landcover_LE', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                    'Aggregate_texture', 'Aquifers', 'bedrock_raster_50m', 'ALC_old']

predict_folder = Path('/dbfs/mnt/lab/unrestricted/KritiM/Predict')
predict_folder.mkdir(parents=True, exist_ok=True)
grid_folder = predict_folder / 'Grid'
grid_folder.mkdir(parents=True, exist_ok=True)
mosaic_folder = predict_folder / 'Mosaic'
mosaic_folder.mkdir(parents=True, exist_ok=True)

written_rasters = []
written_conf_rasters = []
written_prob_rasters = dict()  # key: class index, value: list of rasters

# Only process grid CSVs 
csv_files = [f for f in table_folder.glob('*.csv') if f.stem != 'AAR']
# Get the number of classes from the first ensemble model
n_classes = 12  # Fixed number of output classes
for csv_file in csv_files:
    grid = csv_file.stem
    out_raster_path = grid_folder / f"{grid}_predict_xgb.tif"
    out_conf_raster_path = grid_folder / f"{grid}_confidence_xgb.tif"
    # Check if all class probability rasters already exist for this grid
    all_prob_exist = True
    for c in range(n_classes):
        prob_path = grid_folder / f"{grid}_classprob_{c+1}_xgb.tif"
        if not prob_path.exists():
            all_prob_exist = False
            break
    if all_prob_exist:
        print(f"All class probability rasters for grid {grid} already exist. Skipping prediction and raster generation.")
        # Still add the paths to written_rasters, written_conf_rasters, written_prob_rasters for mosaic step
        if out_raster_path.exists():
            written_rasters.append(str(out_raster_path))
        if out_conf_raster_path.exists():
            written_conf_rasters.append(str(out_conf_raster_path))
        for c in range(n_classes):
            prob_path = grid_folder / f"{grid}_classprob_{c+1}_xgb.tif"
            if prob_path.exists():
                written_prob_rasters.setdefault(c, []).append(str(prob_path))
        continue
    # Find the reference raster that matches the grid name
    ref_raster = next(tifrefdir.glob(f"{grid}_AAR.tif"), None)
    if ref_raster is None:
        print(f"No reference raster for grid {grid}")
        continue
    with rio.open(ref_raster, 'r') as src:
        profile = src.profile.copy()
        profile.update(count=1)
        nodata_value = src.nodatavals[0] if src.nodatavals else -9999
        band = src.read(1)
        grid_profile = {
            'profile': profile,
            'height': src.height,
            'width': src.width,
            'raster_path': str(ref_raster),
            'grid': grid
        }
    out_raster = predict_and_write_ensemble(str(csv_file), ensemble_models, scaler, model_feature_names, categorical_cols, str(grid_folder), grid_profile, nodata_value)
    if out_raster:
        written_rasters.append(str(grid_folder / f"{grid}_predict_xgb.tif"))
        conf_raster = grid_folder / f"{grid}_confidence_xgb.tif"
        if conf_raster.exists():
            written_conf_rasters.append(str(conf_raster))
        for c in range(n_classes):
            prob_path = grid_folder / f"{grid}_classprob_{c+1}_xgb.tif"
            if prob_path.exists():
                written_prob_rasters.setdefault(c, []).append(str(prob_path))

print('Merging all predicted rasters into a mosaic with smooth blending in overlap regions...')
mosaic_output_file = mosaic_folder / 'soil_predict_xgb_mosaic.tif'
if mosaic_output_file.exists():
    print(f"Predicted mosaic raster already exists: {mosaic_output_file}, skipping generation.")
else:
    if written_rasters:
        src_files_to_mosaic = []
        for fp in written_rasters:
            with rio.open(fp) as src:
                # Ensure nodata is set in metadata
                if src.nodata != NODATA_VALUE:
                    profile = src.profile.copy()
                    profile['nodata'] = NODATA_VALUE
                    arr = src.read(1)
                    arr[src.read(1) == src.nodata] = NODATA_VALUE if src.nodata is not None else NODATA_VALUE
                    temp_dir = tempfile.mkdtemp()
                    temp_fp = os.path.join(temp_dir, os.path.basename(fp))
                    with rio.open(temp_fp, 'w', **profile) as dst:
                        dst.write(arr, 1)
                    src_files_to_mosaic.append(rio.open(temp_fp))
                else:
                    src_files_to_mosaic.append(rio.open(fp))
        # Use 'first' method for class predictions (discrete values)
        mosaic, out_trans = rio_merge(src_files_to_mosaic, nodata=NODATA_VALUE, method='first')
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "nodata": NODATA_VALUE,
            "count": 1
        })
        temp_dir = tempfile.mkdtemp()
        temp_mosaic = os.path.join(temp_dir, 'soil_predict_xgb_mosaic.tif')
        with rio.open(temp_mosaic, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)
        shutil.copy2(temp_mosaic, str(mosaic_output_file))
        shutil.rmtree(temp_dir)
        for src in src_files_to_mosaic:
            src.close()
        print(f"Wrote mosaic raster: {mosaic_output_file}")
    else:
        print(f"No predicted rasters found to merge in {grid_folder}")

print('Merging all confidence rasters into a mosaic with smooth blending in overlap regions...')
mosaic_conf_output_file = mosaic_folder / 'soil_confidence_xgb_mosaic.tif'
if mosaic_conf_output_file.exists():
    print(f"Confidence mosaic raster already exists: {mosaic_conf_output_file}, skipping generation.")
else:
    if written_conf_rasters:
        src_files_to_mosaic = []
        for fp in written_conf_rasters:
            with rio.open(fp) as src:
                if src.nodata != NODATA_VALUE:
                    profile = src.profile.copy()
                    profile['nodata'] = NODATA_VALUE
                    arr = src.read(1)
                    arr[src.read(1) == src.nodata] = NODATA_VALUE if src.nodata is not None else NODATA_VALUE
                    temp_dir = tempfile.mkdtemp()
                    temp_fp = os.path.join(temp_dir, os.path.basename(fp))
                    with rio.open(temp_fp, 'w', **profile) as dst:
                        dst.write(arr, 1)
                    src_files_to_mosaic.append(rio.open(temp_fp))
                else:
                    src_files_to_mosaic.append(rio.open(fp))
        # Use 'max' method for confidence values to take the highest confidence in overlap regions
        mosaic, out_trans = rio_merge(src_files_to_mosaic, nodata=NODATA_VALUE, method='max')
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "nodata": NODATA_VALUE,
            "count": 1
        })
        temp_dir = tempfile.mkdtemp()
        temp_mosaic = os.path.join(temp_dir, 'soil_confidence_xgb_mosaic.tif')
        with rio.open(temp_mosaic, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)
        shutil.copy2(temp_mosaic, str(mosaic_conf_output_file))
        shutil.rmtree(temp_dir)
        for src in src_files_to_mosaic:
            src.close()
        print(f"Wrote confidence mosaic raster: {mosaic_conf_output_file}")
    else:
        print(f"No confidence rasters found to merge in {grid_folder}")

print('Merging all class probability rasters into mosaics with smooth blending in overlap regions...')
for c, prob_raster_list in written_prob_rasters.items():
    mosaic_prob_output_file = mosaic_folder / f'soil_classprob_{c+1}_xgb_mosaic.tif'
    if mosaic_prob_output_file.exists():
        print(f"Class probability mosaic raster for class {c+1} already exists: {mosaic_prob_output_file}, skipping generation.")
        continue
    if prob_raster_list:
        src_files_to_mosaic = []
        for fp in prob_raster_list:
            with rio.open(fp) as src:
                if src.nodata != NODATA_VALUE:
                    profile = src.profile.copy()
                    profile['nodata'] = NODATA_VALUE
                    arr = src.read(1)
                    arr[src.read(1) == src.nodata] = NODATA_VALUE if src.nodata is not None else NODATA_VALUE
                    temp_dir = tempfile.mkdtemp()
                    temp_fp = os.path.join(temp_dir, os.path.basename(fp))
                    with rio.open(temp_fp, 'w', **profile) as dst:
                        dst.write(arr, 1)
                    src_files_to_mosaic.append(rio.open(temp_fp))
                else:
                    src_files_to_mosaic.append(rio.open(fp))
        # Use 'max' method for probability values to take the highest probability in overlap regions
        mosaic, out_trans = rio_merge(src_files_to_mosaic, nodata=NODATA_VALUE, method='max')
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "nodata": NODATA_VALUE,
            "count": 1
        })
        temp_dir = tempfile.mkdtemp()
        temp_mosaic = os.path.join(temp_dir, f'soil_classprob_{c+1}_xgb_mosaic.tif')
        with rio.open(temp_mosaic, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)
        shutil.copy2(temp_mosaic, str(mosaic_prob_output_file))
        shutil.rmtree(temp_dir)
        for src in src_files_to_mosaic:
            src.close()
        print(f"Wrote class probability mosaic raster for class {c+1}: {mosaic_prob_output_file}")
    else:
        print(f"No class probability rasters found for class {c+1} to merge in {grid_folder}")

print('Deleting per-grid confidence and class probability rasters...')
# The following deletion steps are commented out to preserve all files in the Grid folder:
# for fp in written_conf_rasters:
#     try:
#         os.remove(fp)
#         print(f"Deleted {fp}")
#     except Exception as e:
#         print(f"Could not delete {fp}: {e}")
# for prob_raster_list in written_prob_rasters.values():
#     for fp in prob_raster_list:
#         try:
#             os.remove(fp)
#             print(f"Deleted {fp}")
#         except Exception as e:
#             print(f"Could not delete {fp}: {e}")
# for f in grid_folder.glob("*_classprob_*_xgb.tif"):
#     if "mosaic" not in f.name:
#         try:
#             os.remove(f)
#             print(f"Deleted {f}")
#         except Exception as e:
#             print(f"Could not delete {f}: {e}")
# for f in grid_folder.glob("*_confidence_xgb.tif"):
#     if "mosaic" not in f.name:
#         try:
#             os.remove(f)
#             print(f"Deleted {f}")
#         except Exception as e:
#             print(f"Could not delete {f}: {e}")

end = time.time()
timetaken = convert(end-start)
print('Time taken for processing: ', timetaken)

# --- CSV column check against raster files in each grid subdirectory ---
print("\nChecking that CSV columns match raster files in each grid subdirectory...")
for folder in subdirectories:
    grid = folder.name
    csv_path = table_folder / f"{grid}.csv"
    if not csv_path.exists():
        print(f"CSV for grid {grid} not found: {csv_path}")
        continue
    # Get raster file stems (without extension)
    raster_files = list(folder.glob('*.tif'))
    raster_stems = set(f.stem for f in raster_files)
    # Read CSV columns
    df = pd.read_csv(csv_path, nrows=1)  # Only need header
    csv_columns = set(df.columns) - {'EAST', 'NORTH', 'TILE'}
    # Compare
    missing_in_csv = raster_stems - csv_columns
    extra_in_csv = csv_columns - raster_stems
    if not missing_in_csv and not extra_in_csv:
        print(f"Grid {grid}: All raster files are present as columns in the CSV.")
    else:
        if missing_in_csv:
            print(f"Grid {grid}: Raster files missing as columns in CSV: {sorted(missing_in_csv)}")
        if extra_in_csv:
            print(f"Grid {grid}: CSV columns not found as raster files: {sorted(extra_in_csv)}")
