# 🌱 Soil Mapping for England

This repository contains scripts for soil data cleaning, classification, raster preprocessing, and nationwide prediction using machine learning.

---

## 📁 Repository Structure

### `scripts/soil_data_qa.py`
Cleans the raw soil profile dataset and extracts the attributes required for soil classification.

### `scripts/ml_train_class_balance_weighted.py`
Trains the XGBoost soil classification model, using class balancing and weighting to address imbalanced classes.

### `scripts/align_raster.py`
Aligns all predictor rasters to a common reference raster.  
- Place all predictor rasters in the directory defined by `path`.  
- Set `reference` as the raster used for alignment.

### `scripts/Raster_grids.py`
Splits large predictor rasters into smaller grid tiles for efficient computation.  
- Store input rasters in the folder specified by `pathRaster` (Line 24).

### `scripts/predict_model.py`
Generates national-scale soil class predictions using the trained model.

---

## 🗺️ Interactive Map

Explore the results using the interactive Juxtapose viewer:  
👉 **[Before/After Map](https://kmu00009.github.io/SoilMapping/juxtapose.html)**

---


