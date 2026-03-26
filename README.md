# 🌱 Soil Mapping for England

This repository contains scripts for soil data cleaning, classification, raster preprocessing, and nationwide prediction using machine learning.

---

## 📁 Repository Structure

### `scripts/soil_data_qa.py`
Cleans the raw soil profile dataset and extracts the attributes required for soil classification.

### `scripts/dataPreparation.ipynb`

**Purpose:**  
Generates labelled soil samples for model training.

### `scripts/align_raster.py`
Aligns all predictor rasters to a common reference raster.  
- Place all predictor rasters in the directory defined by `path`.  
- Set `reference` as the raster used for alignment.

**Notebook structure:**
- **Cell 1:** Loads the cleaned soil profile output from `soil_data_qa.py` and classifies the profiles into soil types.  
- **Cell 2:** Uses soil polygon shapefiles to extract predictor raster values at those locations, producing labelled soil samples.

**Output:**  
A dataset linking soil types with predictor variables.

### `scripts/align_raster.py`
Aligns all predictor rasters to a common reference raster.  
- Place all predictor rasters in the directory defined by `path`.  
- Set `reference` as the raster used for alignment.

### `scripts/ml_train_class_balance_weighted.py`
Trains the XGBoost soil classification model, using class balancing and weighting to address imbalanced classes.


### `scripts/Raster_grids.py`
Splits large predictor rasters into smaller grid tiles for efficient computation.  
- Store input rasters in the folder specified by `pathRaster` (Line 24).

### `scripts/predict_model_seed.py`
Generates national-scale soil class predictions using the trained model.



