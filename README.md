# 🌱 Soil Mapping for England

This repository contains scripts for soil data cleaning, classification, raster preprocessing, and nationwide prediction using machine learning.

---

## 📁 Repository Structure

### `scripts/soil_data_qa.py`
Cleans the raw soil profile dataset and extracts the attributes required for soil classification.

### `scripts/align_raster.py`
Aligns all predictor rasters to a common reference raster.  
- Place all predictor rasters in the directory defined by `path`.  
- Set `reference` as the raster used for alignment.

### `scripts/explore predict data.ipynb`
using the script under the cell 'Additional training samples from Peat data', generate the trainingSample.csv file for model training. It uses two point shape files: soil_class_12.shp and Peat_Matthew.shp nd the predictor rasters to extract the labelled samples.


### `scripts/ml_train_class_balance_weighted.py`
Trains the XGBoost soil classification model, using class balancing and weighting to address imbalanced classes.


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


