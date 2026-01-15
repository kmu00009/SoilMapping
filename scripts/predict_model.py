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

def predictClass_chunked(df, classifier, scaler, feature_names, chunk_size=100000):
    class_preds = []
    confidence_preds = []
    nan_masks = []
    for chunk in split_dataframe(df, chunk_size):
        nan_mask = chunk.isna().any(axis=1)
        chunk_filled = chunk.fillna(0)
        chunk_pred = chunk_filled[feature_names]
        dtest = xgb.DMatrix(chunk_pred, enable_categorical=True)
        class_values = classifier.predict(dtest)
        temp_model_file = 'temp_model.json'
        classifier.save_model(temp_model_file)
        sk_classifier = xgb.XGBClassifier()
        sk_classifier.load_model(temp_model_file)
        prob_values = sk_classifier.predict_proba(chunk_pred)
        confidence_values = np.max(prob_values, axis=1)
        os.remove(temp_model_file)
        class_preds.append(class_values + 1)  # 1-based
        confidence_preds.append(confidence_values)
        nan_masks.append(nan_mask)
    class_preds = np.concatenate(class_preds)
    confidence_preds = np.concatenate(confidence_preds)
    nan_mask_full = pd.concat(nan_masks).values if nan_masks else np.array([])
    class_preds[nan_mask_full] = np.nan
    confidence_preds[nan_mask_full] = np.nan
    return class_preds, confidence_preds

def predict_and_write(grid_csv, classifier, scaler, model_feature_names, categorical_cols, output_predict_folder, grid_profile, nodata_value):
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
    class_preds, confidence_preds = predictClass_chunked(df_scaled_select, classifier, scaler, model_feature_names)
    df.loc[valid_mask, 'Class'] = class_preds
    df.loc[valid_mask, 'Confidence'] = confidence_preds
    df.to_csv(grid_csv, index=False)
    band_shape = (grid_profile['height'], grid_profile['width'])
    class_array = np.full(df.shape[0], np.nan)
    class_array[valid_mask] = class_preds
    class_array = class_array.reshape(band_shape)
    if nodata_value is not None:
        with rio.open(grid_profile['raster_path'], 'r') as src:
            band = src.read(1)
        class_array[band == nodata_value] = nodata_value
    
    # Write to local temp directory first to avoid DBFS FUSE issues
    temp_dir = tempfile.mkdtemp()
    temp_raster = os.path.join(temp_dir, f"{grid_profile['grid']}_predict_xgb.tif")
    profile = grid_profile['profile']
    with rio.open(temp_raster, 'w', **profile) as dst:
        dst.write(class_array.astype(np.float32), 1)
    
    # Copy to final destination
    out_raster = os.path.join(output_predict_folder, f"{grid_profile['grid']}_predict_xgb.tif")
    shutil.copy2(temp_raster, out_raster)
    
    # Clean up temp file
    shutil.rmtree(temp_dir)
    
    print(f"Wrote predicted raster: {out_raster}")
    return out_raster

# Main execution
start = time.time()
print('Checking Table folder for existing CSVs for each grid...')

# pathtogrids = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID')
table_folder = Path('/dbfs/mnt/lab/unrestricted/KritiM/Table')
table_folder.mkdir(parents=True, exist_ok=True)
subdirectory = Path('/dbfs/mnt/lab/unrestricted/KritiM/GRID/AAR')

expected_csvs = {}
tif_files = list(subdirectory.glob("*_AAR.tif"))
for tif_file in tif_files:
    prefix = tif_file.stem.rsplit("_AAR", 1)[0]
    expected_csvs[prefix] = table_folder / f"{prefix}.csv"

missing_csvs = [grid for grid, csv_path in expected_csvs.items() if not csv_path.exists()]

if not missing_csvs:
    print(f"All Table CSVs already exist in {table_folder}. Skipping Table generation step.")
else:
    print(f"Missing Table CSVs for grids: {missing_csvs}. Generating only missing ones...")
    for folder in subdirectories:
        grid = folder.name
        if grid not in missing_csvs:
            continue
        files = list(folder.glob('*.tif'))
        if not files:
            print(f"No .tif files in {folder}")
            continue
        # Group rasters by tile prefix (e.g., NT_NE, NT_SE, etc.)
        tile_groups = {}
        for file in files:
            # Extract tile prefix (e.g., NT_NE) from filename
            stem = file.stem
            parts = stem.split('_')
            if len(parts) < 2:
                print(f"Warning: Unexpected raster name format: {file.name}. Skipping.")
                continue
            tile_prefix = '_'.join(parts[:2])
            tile_groups.setdefault(tile_prefix, []).append(file)
        tile_dfs = []
        for tile, tile_files in tile_groups.items():
            # Check all rasters in this tile have the same shape
            shapes = []
            for f in tile_files:
                with rio.open(f, 'r') as src:
                    shapes.append(src.shape)
            if len(set(shapes)) != 1:
                print(f"Warning: Not all rasters in tile {tile} for grid {grid} have the same shape. Skipping this tile.")
                continue
            # All shapes are the same, proceed
            with rio.open(tile_files[0], 'r') as src:
                height, width = src.height, src.width
                rows, cols = np.meshgrid(
                    np.arange(height),
                    np.arange(width),
                    indexing="ij"
                )
                xs, ys = rio.transform.xy(src.transform, rows, cols)
                tile_df = pd.DataFrame({
                    'EAST': np.array(xs).ravel(),
                    'NORTH': np.array(ys).ravel()
                })
            for f in tile_files:
                var = f.stem
                with rio.open(f, 'r') as src:
                    data = src.read(1)
                    tile_df[var] = data.ravel()
            tile_df['TILE'] = tile  # Optionally add tile label
            tile_dfs.append(tile_df)
        if tile_dfs:
            df = pd.concat(tile_dfs, ignore_index=True)
            df.to_csv(table_folder / f"{grid}.csv", index=False)
            print(f"Wrote {table_folder / f'{grid}.csv'} with {len(df)} rows.")
        else:
            print(f"No valid tiles found for grid {grid}. No CSV written.")

print('Loading model and scaler...')
pathC = Path(os.environ.get('CLASSIFICATION_DIR', '/dbfs/mnt/lab/unrestricted/KritiM/classification/'))
model_path = Path(os.environ.get('MODEL_PATH', str(pathC / 'model_10_features.joblib')))
scaler_path = Path(os.environ.get('SCALER_PATH', str(pathC / 'scaler.joblib')))
best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_feature_names = get_model_feature_names(model_path)
if model_feature_names is None:
    model_feature_names = scaler.feature_names_in_.tolist()

categorical_cols = ['Landcover_LE', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                    'Aggregate_texture', 'Aquifers', 'bedrock_raster_50m', 'ALC_old']

predict_folder = Path('/dbfs/mnt/lab/unrestricted/KritiM/Predict')
predict_folder.mkdir(parents=True, exist_ok=True)

# Only process grid CSVs (filter out AAR.csv and any non-grid files)
csv_files = [f for f in table_folder.glob('*.csv') if f.stem != 'AAR']
# print('Filtered CSV files:', csv_files)
written_rasters = []
for csv_file in csv_files:
    grid = csv_file.stem
    
    # Check if output raster already exists
    out_raster_path = predict_folder / f"{grid}_predict_xgb.tif"
    if out_raster_path.exists():
        print(f"Predicted raster already exists for grid {grid}. Skipping.")
        written_rasters.append(str(out_raster_path))
        continue
    
    # Find the reference raster that matches the grid name
    ref_raster = next(subdirectory.glob(f"{grid}_AAR.tif"), None)
    # print(f"Processing grid: {grid}, Reference raster: {ref_raster}")
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
    try:
        out_raster = predict_and_write(str(csv_file), best_model, scaler, model_feature_names, categorical_cols, str(predict_folder), grid_profile, nodata_value)
        if out_raster:
            written_rasters.append(out_raster)
    except Exception as e:
        print(f"Error writing raster for grid {grid}: {e}")
        import traceback
        traceback.print_exc()

print('Merging all predicted rasters into a mosaic...')
if written_rasters:
    # Open and close files one at a time to avoid OOM
    src_files_to_mosaic = []
    for fp in written_rasters:
        src = rio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = rio_merge(src_files_to_mosaic, nodata=-9999)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "nodata": -9999,
        "count": 1
    })
    
    # Write mosaic to local temp directory first to avoid DBFS FUSE issues
    temp_dir = tempfile.mkdtemp()
    temp_mosaic = os.path.join(temp_dir, 'soil_predict_xgb_mosaic.tif')
    with rio.open(temp_mosaic, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)
    
    # Copy to final destination
    mosaic_output_file = predict_folder / 'soil_predict_xgb_mosaic.tif'
    shutil.copy2(temp_mosaic, str(mosaic_output_file))
    
    # Clean up temp file
    shutil.rmtree(temp_dir)
    
    # Close all opened files
    for src in src_files_to_mosaic:
        src.close()
    print(f"Wrote mosaic raster: {mosaic_output_file}")
else:
    print(f"No predicted rasters found to merge in {predict_folder}")

end = time.time()
timetaken = convert(end-start)
print('Time taken for processing: ', timetaken)
