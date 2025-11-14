# -*- coding: utf-8 -*-
"""
The soil sample data given to UKCEH and new data created by NE are used to create the merged,
quality controlled soil sample data using only 'WETCLASS', 'CACO3', 'X_BTMDEPTH', 'X_TEXTURE', 
and '1_TOTSTONE' columns. These columns were used by UKCEH to create 14 soil classes.
"""

import pandas as pd
import numpy as np
from soil_data_qa_functions import process_munsell_csv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def QWETNESS(df):
    df = df.dropna(subset=['WETCLASS'])
    df.loc[:, 'WETCLASS']  = np.where(df['WETCLASS']==9, np.nan, df['WETCLASS']+1)
    return df

# refine CACO3 columns: set all values other than 'Y' or 'y' as 'N'
def QCACO3(df):
    for column in df.columns:
        if 'CACO3' in column:
            # Convert to string to safely handle NaN, then upper-case it
            df.loc[:, column] = (
                df[column]
                .astype(str)
                .str.upper()
                .apply(lambda x: 'Y' if x == 'Y' else 'N')
                .astype('string')   # explicitly cast to string dtype
            )
    return df

# prepare DEPTH_TOTAL column: max of depth for all horizons
def QBTMDEPTH(df):
    df = df.copy()  # ensures we’re not modifying a view
    depth_columns = [col for col in df.columns if 'DEPTH' in col.upper()]
    
    # Convert each depth column to numeric, forcing errors to NaN
    for col in depth_columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
    df.loc[:, 'DEPTH_TOTAL'] = df[depth_columns].max(axis=1)
    return df

# There is no 'TOTSTONE' 
def QSTONE(df):
    df.loc[:,'TOTSTONE'] = df['1_TOTSTONE']
    return df

# prepare the texture columns using texture look up values
def QTEXTURE(df, dfLU, name):
    texture_lookup = dict(
        zip(
            dfLU[dfLU['CATEGORY'] == 'TEXTURE']['LOWER'],
            dfLU[dfLU['CATEGORY'] == 'TEXTURE']['SOIL_CLASS_CRITERIA'],
        )
    )

    # Step 2: identify lookup entries where SOIL_CLASS_CRITERIA is NaN
    null_textures = set(
        dfLU.loc[
            (dfLU['CATEGORY'] == 'TEXTURE') &
            (dfLU['SOIL_CLASS_CRITERIA'].isna()),
            'LOWER'
        ]
    )

    texture_columns = [col for col in df.columns if '_TEXTURE' in col]
    print(texture_columns)

    # Map each texture column to a new converted column using the lookup
    for col in texture_columns:
        new_col = col.replace('_TEXTURE', '_TEXTURE_CLASS')
        df[new_col] = df[col].map(texture_lookup)
    
    # Step 2: Find unmatched texture values    
    all_textures = pd.unique(df[texture_columns].values.ravel())
    unmatched_textures = [
        t for t in all_textures
        if pd.notna(t) and (t not in texture_lookup or t in null_textures)
    ]
    print("Unmatched textures found:", unmatched_textures)
    print("Count:", len(unmatched_textures))
    
    # Count occurrences of each unmatched texture in the full dataframe
    from collections import Counter
    
    # Flatten all values from relevant columns
    flattened_textures = df[texture_columns].values.ravel()
    # Keep only unmatched values and notna
    unmatched_counts = Counter(t for t in flattened_textures if pd.notna(t) and t in unmatched_textures)
    
    # Get total number of rows (for %)
    total_rows = len(df)
    
    # Step 3: Create new rows for unmatched values, with counts and percentages
    new_df = pd.DataFrame({
        'CATEGORY': ['TEXTURE'] * len(unmatched_textures),
        'LOWER': unmatched_textures,
        'SOIL_CLASS_CRITERIA': [None] * len(unmatched_textures),
        'COUNT': [unmatched_counts.get(t, 0) for t in unmatched_textures],
        'PERCENTAGE': [round((unmatched_counts.get(t, 0) / total_rows) * 100, 2) for t in unmatched_textures]
    })
    
    # Step 4: Append and save
    new_df.to_csv('../Data/UKCEH/kriti/ALC_unmatched_texture_stat' + name + '.csv', index=False)
    
    print("Lookup table updated and saved as 'ALC_SOILS_LOOKUP_extended_ER_stat.csv'")    
    
    return df
    

dataU = "../Data/UKCEH/kriti/raw/SOIL_UKCEH.csv"
dataM = "../Data/UKCEH/kriti/raw/SOIL_MATTHEW.csv"
Munsell_LU = "../Data/Matthew/Munsell_rgb_intensity.csv"
# convert Munsell codes to RGB and intensity values
# process_munsell_csv(dataM, Munsell_LU)
# process_munsell_csv(dataU, Munsell_LU)

# Read input CSV
soilU = pd.read_csv(dataU, low_memory=False)
soilM = pd.read_csv(dataM, low_memory=False)
TextureLU = pd.read_csv('../Data/UKCEH/kriti/ALC_SOILS_LOOKUP_ER.csv', encoding='latin1')


# read the size of the data
print(f'initial size of raw UKCEH training data: {soilU.shape}')
print(f'initial size of raw new training data: {soilM.shape}')
print(f'initial size of raw sample size: {soilU.shape[0]+soilM.shape[0]}')

train_data = [soilU, soilM]

name = ['UKCEH', 'MATTHEW']

processed_data = []
for i in range(len(train_data)):    
    df = train_data[i]
    df = QWETNESS(df)
    df = QCACO3(df)
    df = QBTMDEPTH(df)
    df = QSTONE(df)
    df = QTEXTURE(df, TextureLU, name[i])
    df = df[['DEPTH_TOTAL', '1_CACO3', 'WETCLASS', 'TOTSTONE', '1_TEXTURE_CLASS', '1_Intensity', '1_R', '1_G', 
             '1_B', 'EASTING', 'NORTHING']]
    df.to_csv('../Data/UKCEH/kriti/SOIL_QC_' + name[i] + '.csv', index=False)
    processed_data.append(df)
    

df_merged = pd.concat(processed_data, axis=0, ignore_index=True)
df_merged.to_csv('../Data/UKCEH/kriti/SOIL_data_Final.csv', index=False)
   

