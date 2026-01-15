# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:47:00 2024

@author: kriti.mukherjee
functions for classification using machine learning algorithms
"""

import pandas as pd
import numpy as np
import re
import time
import os
import geopandas as gpd
from osgeo import ogr
import xgboost as xgb

'''
def extractExtent(shape, outpath):
    shape = os.path.abspath(shape)    
    # outpath = os.path.abspath(outpath)
    inDataSource = ogr.Open(shape)
    inLayer = inDataSource.GetLayer(0)    
    Gla_extents = open(os.path.join(outpath, "Extent.txt"), 'w')
    Gla_paths = open(os.path.join(outpath, "ExtPath.txt"), 'w')
    
    for i in range(inLayer.GetFeatureCount()):
        feature = inLayer.GetFeature(i)
        ID = feature.GetField("CODE50")
        geometry = feature.GetGeometryRef()
        if geometry is None:
            print(f"Warning: Feature {ID} has no valid geometry. Skipping...")
            continue 
        minLong,maxLong,minLat,maxLat = geometry.GetEnvelope()       
        Gla_extents.write("te_")        
        Gla_extents.write(ID)
        Gla_paths.write(ID)        
        Gla_extents.write(" = ")
        Gla_paths.write(" = " + outpath + "/GridsPoly/CODE50_")
        Gla_paths.write(ID)
        Gla_paths.write(".shp\n")        
        Gla_extents.write("' -te ")        
        Gla_extents.write(str(minLong))        
        Gla_extents.write(" ")        
        Gla_extents.write(str(minLat))        
        Gla_extents.write(" ")        
        Gla_extents.write(str(maxLong))       
        Gla_extents.write(" ")        
        Gla_extents.write(str(maxLat))        
        Gla_extents.write(" '\n")

    Gla_extents.close()
    Gla_paths.close()
    print("All Done!")


def ReadGlaExtent(txtfile):
        inFile = open(txtfile,'r')
        text = inFile.readlines()
        ID = []
        value = []
        lines = 0
        for i in text:
            if i == "":
                break
            else:
                ID.append(i[3:8])
                value.append(i[12:-2])
                lines += 1
        return ID, value, lines

def ReadGlaPoly(txtfile):
        inFile = open(txtfile,'r')
        text = inFile.readlines()
        ID = []
        value = []
        lines = 0
        for i in text:
            if i == "":
                break
            else:
                ID.append(i[0:5])
                value.append(i[8:-1])
                lines += 1
        return ID, value, lines

def splitShape(shape, outdir):
    gdf = gpd.read_file(shape)
    
    attribute_column = 'CODE50'
    # iterate through each polygon and save them separately
    for i, row in gdf.iterrows():
        name = str(row[attribute_column]).replace(" ", "_")
        
        sgdf = gpd.GeoDataFrame([row], columns=gdf.columns, crs=gdf.crs)
        outpath = os.path.join(outdir, f'CODE50_{name}.shp')
        sgdf.to_file(outpath)
'''
def split_dataframe(df, chunk_size):
    """Split a DataFrame into smaller DataFrames of a specified chunk size."""
    num_chunks = len(df) // chunk_size + int(len(df) % chunk_size != 0)
    return [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def predictClass(infile, outpath, i, classifier, scaler, feature_names):
    import os
    try:
        # Load data
        df = pd.read_csv(infile)
        print(f"Processing chunk {i}")
        
        # Handle NaNs (as in training, you dropped them, but here you fill with 0)
        nan_mask = df.isna().any(axis=1)
        df_filled = df.fillna(0)

        # Reorder columns to match feature_names
        df_pred = df_filled[feature_names]

        # Convert to DMatrix for class prediction
        dtest = xgb.DMatrix(df_pred, enable_categorical=True)
        
        # Predict class values
        class_values = classifier.predict(dtest)
        
        # Save model to temporary file
        temp_model_file = 'temp_model.json'
        classifier.save_model(temp_model_file)
        
        # Verify that the model file was created
        if not os.path.exists(temp_model_file):
            raise FileNotFoundError(f"Temporary model file {temp_model_file} was not created.")
        
        # Use scikit-learn wrapper for probabilities
        sk_classifier = xgb.XGBClassifier()
        try:
            sk_classifier.load_model(temp_model_file)
        except Exception as e:
            raise ValueError(f"Failed to load model into XGBClassifier: {e}")
        
        # Compute probabilities
        prob_values = sk_classifier.predict_proba(df_pred)
        
        # Compute confidence as the maximum probability for each sample
        confidence_values = np.max(prob_values, axis=1)
        print(f"Chunk {i} confidence range: {confidence_values.min():.3f} to {confidence_values.max():.3f}")

        # Add predictions to output DataFrame
        df_filled['Class'] = class_values + 1  # Adjust to 1-based labels
        df_filled['Confidence'] = confidence_values
        
        # Restore NaN for rows that had NaN in original data
        df_filled.loc[nan_mask, ['Class', 'Confidence']] = np.nan

        # Save to CSV
        output_file = os.path.join(outpath, f'class_{i}.csv')
        df_filled.to_csv(output_file, index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Clean up temporary model file
        if os.path.exists(temp_model_file):
            os.remove(temp_model_file)
            print(f"Removed temporary model file: {temp_model_file}")


def extract_index(filename):
    # Extract the numerical index from the filename using a regular expression
    match = re.search(r'class_(\d+)\.csv', filename)
    return int(match.group(1)) if match else -1

    