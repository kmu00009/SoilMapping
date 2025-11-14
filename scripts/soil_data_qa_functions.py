# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 10:28:51 2025

@author: km000009
"""

import pandas as pd
import numpy as np
import re

def parse_munsell_code(code):
    """Convert Munsell code like '10YR44 00' to '10.0YR-4-4', using first six characters."""
    if not isinstance(code, str) or not code or code.lower() in ['nan', '0', '00', '10yr', '10yr42', '10yr31']:
        return None
    try:
        # Take first six characters, strip spaces, and normalize case
        code = code[:6].strip().upper()
        # Match pattern: hue (e.g., '10YR', '5B') followed by two digits (e.g., '44')
        match = re.match(r"(\d*\.?\d*)([A-Z]+)?(\d{2})?", code)
        if not match or not match.group(3):
            return None
        num_part, letter_part, value_chroma = match.groups()
        if not letter_part or not value_chroma or len(value_chroma) < 2:
            return None
        # Extract value and chroma from value_chroma (e.g., '44' -> value '4', chroma '4')
        value = value_chroma[0]
        chroma = value_chroma[1]
        # Convert numerical part of hue according to specified rules
        hue_map = {
            '10': '10.0',
            '25': '2.5',
            '75': '7.5',
            '05': '5.0'
        }
        num_part = num_part.rstrip('.')
        num_part = hue_map.get(num_part, num_part)
        # If num_part is still an integer (e.g., '5'), add '.0'
        try:
            if float(num_part).is_integer():
                num_part = f"{float(num_part):.1f}"
        except ValueError:
            return None
        # Combine hue (e.g., '10.0YR')
        hue = f"{num_part}{letter_part}"
        return f"{hue}-{value}-{chroma}"
    except Exception:
        return None

def load_lookup_table(lookup_file):
    """Load the Munsell lookup table CSV into a dictionary with intensity and RGB values."""
    try:
        df = pd.read_csv(lookup_file)
        # Create dictionary mapping Munsell code to intensity and RGB values
        lookup = {
            row['Munsell']: {
                'intensity': row['Intensity'],
                'R': row['R'] if 'R' in df.columns else None,
                'G': row['G'] if 'G' in df.columns else None,
                'B': row['B'] if 'B' in df.columns else None
            } for _, row in df.iterrows()
        }
        return lookup
    except Exception as e:
        print(f"Error loading lookup table: {e}")
        return {}

def munsell_to_rgb_intensity(munsell_code, lookup_table):
    """Convert a Munsell code to intensity and RGB values using the lookup table; return blank if input is empty."""
    # Check if input is empty (NaN, empty string, None)
    if pd.isna(munsell_code) or munsell_code == '' or munsell_code is None:
        return {'intensity': '', 'R': '', 'G': '', 'B': ''}
    parsed_code = parse_munsell_code(munsell_code)
    if not parsed_code:
        return {'intensity': f"Error: Invalid Munsell code format for '{munsell_code}'", 'R': '', 'G': '', 'B': ''}
    entry = lookup_table.get(parsed_code)
    if entry is None:
        return {
            'intensity': f"Error: {munsell_code} (parsed as {parsed_code}) not found in lookup table",
            'R': '', 'G': '', 'B': ''
        }
    return {
        'intensity': entry['intensity'],
        'R': entry['R'] if entry['R'] is not None else '',
        'G': entry['G'] if entry['G'] is not None else '',
        'B': entry['B'] if entry['B'] is not None else ''
    }

def process_munsell_csv(input_file, lookup_file):
    """Process CSV with Munsell codes, adding intensity and RGB columns."""
    try:
        lookup_table = load_lookup_table(lookup_file)
        if not lookup_table:
            print("Failed to load lookup table. Exiting.")
            return
        
        df = pd.read_csv(input_file, low_memory=False)
        
        munsell_columns = [col for col in df.columns if col.endswith('_COLOUR')]
        if not munsell_columns:
            print("Error: No columns ending with '_COLOUR' found in input CSV")
            return
        
        missing_codes = set()  # Store missing parsed codes
        
        for col in munsell_columns:
            intensity_col = col.replace('_COLOUR', '_Intensity')
            r_col = col.replace('_COLOUR', '_R')
            g_col = col.replace('_COLOUR', '_G')
            b_col = col.replace('_COLOUR', '_B')
            
            def convert_and_track(x):
                result = munsell_to_rgb_intensity(x, lookup_table)
                # Check if result['intensity'] is a string and starts with 'Error'
                if isinstance(result['intensity'], str) and result['intensity'].startswith('Error'):
                    if 'parsed as' in result['intensity']:
                        missing_codes.add(result['intensity'].split('parsed as ')[1].split(')')[0])
                    return pd.Series([np.nan, np.nan, np.nan, np.nan], index=[intensity_col, r_col, g_col, b_col])
                return pd.Series(
                    [result['intensity'], result['R'], result['G'], result['B']],
                    index=[intensity_col, r_col, g_col, b_col]
                )
            
            # Apply conversion and assign results to new columns
            df[[intensity_col, r_col, g_col, b_col]] = df[col].apply(convert_and_track)
        
        df.to_csv(input_file, index=False)
        print(f"Updated CSV saved to {input_file}")
        
        if missing_codes:
            missing_list = sorted(missing_codes)
            print("\nMissing Munsell codes (parsed format):")
            pd.Series(missing_list, name="Missing_Munsell_Code").to_csv(
                "../Data/trainingData/raw/missing_munsell_codes.csv", index=False, header=True)
            print("\nSaved missing codes to missing_munsell_codes.csv")
        else:
            print("\n✅ All Munsell codes found in lookup table.")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")