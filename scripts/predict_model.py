# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:25:31 2025

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
from predictFunctions import split_dataframe, predictClass, extract_index

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

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
    """
    Extract feature names from an XGBoost model saved in a .joblib file.
    
    Args:
        model_path (str): Path to the model .joblib file.
    
    Returns:
        list: List of feature names used by the model, or None if not available.
    """
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

def predict(grid):
    # Load the pre-trained model and scaler
    try:
        model_path = 'classification/model_9_features.joblib'
        scaler_path = 'classification/scaler.joblib'
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
    dftrain = pd.read_csv('classification/trainingSample.csv')
    mode_values = {col: dftrain[col].mode()[0] for col in categorical_cols if col in dftrain.columns}
    mean_values = dftrain.select_dtypes(include='number').mean()
    traincols = dftrain.columns.tolist()
    pathtogrids = Path('../../Data/predictData/cofactors_used/GRID/')
    subdirectories = [subdir for subdir in pathtogrids.iterdir() if subdir.is_dir()]
    subdirectories.sort()
    
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
    
    # Write dataframe to CSV
    outpath = Path('../../Data/predictData/cofactors_used')
    outdir = outpath / 'Table'
    
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {outdir}: {e}")
        raise
    df.to_csv(outdir / (grid + '.csv'), index=False)
    
    # Read one of the raster grid files for profile information
    pathraster = Path('../../Data/predictData/cofactors_used/GRID/AAR')
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
    pathdata = Path('../../Data/predictData/cofactors_used/Table')
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
        
        tmp = Path('FinalOutputs/Predict') / grid
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
        with rio.open(Path('FinalOutputs/Predict') / f'{grid}_predict_xgb.tif', 'w', **profile) as dst:
            dst.write(S_class.astype(rio.float32), 1)
        
        # Save the confidence image
        with rio.open(Path('FinalOutputs/Predict') / f'{grid}_confidence_xgb.tif', 'w', **profile) as dst:
            dst.write(S_confidence.astype(rio.float32), 1)
        
        # Delete the temporary folder
        shutil.rmtree(tmp)
    except Exception as e:
        print(f'Error predicting for {grid}: {e}')
        return

# Main execution
start = time.time()
print('Extract extents from the grids...')
path = "../../Data"
grid = os.path.join(path, "England_ALC50_GRID.shp")
extractExtent(grid, path)

print('Prepare shapefile for each grid....')
grid = os.path.join("../../Data/England_ALC50_GRID.shp")
outgridpath = '../../Data/GridsPoly'
os.makedirs(outgridpath, exist_ok=True)
# split the shape file in different polygons
splitShape(grid, outgridpath)


print('Split the input rasters into grids...')
# path to the tif file which needs to be split
pathRaster = '../../Data/predictData/cofactors_used'

# path to the shapefiles extents
gridPath = '../../Data/ExtPath.txt'
extents = '../../Data/Extent.txt'
'''
Grids = ReadGlaExtent(extents)  

polygons = ReadGlaPoly(gridPath)

# read the input rasters to split
data = glob.glob(pathRaster + '/*.tif')
print('input rasters to split: ', data)

# split the grids for all input rasters
for raster in data:
    print(f'splitting grids for {raster}')
    suffix = os.path.basename(raster)[:-4] 
    outgrid = os.path.join(pathRaster, 'GRID', suffix)
    os.makedirs(outgrid, exist_ok=True)
    
    for i in range(len(polygons[1])):
        print('polygons[1][i]: ', polygons[1][i])
        print('polygons[0][i]: ', polygons[0][i])
        
        # outgrid = os.path.join(outpath,polygons[0][i])
        # os.makedirs(outgrid, exist_ok=True)
        print('outgrid: ', outgrid)
        
        print('gdalwarp -overwrite -cutline ' + polygons[1][i] + ' ' + Grids[1][i] + ' -r near ' + raster + ' ' +
                           outgrid + '/' +  polygons[0][i] + '_' + suffix + '.tif'  + ' ' + '-dstnodata' + ' ' + '-9999' )
        
        os.system('gdalwarp -overwrite -cutline ' + polygons[1][i] + ' ' + Grids[1][i] + ' -r near ' + raster + ' ' +
                           outgrid + '/' +  polygons[0][i] + '_' + suffix + '.tif'  + ' ' + '-dstnodata' + ' ' + '-9999')
    
'''    
path = Path('../../Data/predictData/cofactors_used/GRID/AAR/')
files = list(path.glob('*.tif'))
for file in files:
    grid = file.name[:5]
    output_file = Path('FinalOutputs/Predict') / f'{grid}_predict_xgb.tif'
    if not output_file.exists():
        predict(grid)

print('Merging the grids together...')
directory = 'FinalOutputs/Predict'
os.makedirs(directory, exist_ok=True)
# Merge class rasters
filenames = glob.glob(os.path.join(directory, '*_predict_xgb.tif'))
output_file = 'FinalOutputs/soil_predict_xgb_.tif'
command = f"gdal_merge.py -a_nodata -9999 -o \"{output_file}\" " + " ".join([f"\"{file}\"" for file in filenames])
os.system(command)

# Merge confidence rasters
confidence_filenames = glob.glob(os.path.join(directory, '*_confidence_xgb.tif'))
confidence_output_file = 'FinalOutputs/soil_confidence_xgb_.tif'
command = f"gdal_merge.py -a_nodata -9999 -o \"{confidence_output_file}\" " + " ".join([f"\"{file}\"" for file in confidence_filenames])
os.system(command)

end = time.time()
timetaken = convert(end-start)
print('Time taken for processing: ', timetaken)