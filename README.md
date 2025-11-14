# SoilMapping
Codes to create soil classification and mapping for England

scripts/soil_data_qa.py: Use this script to clean the soil profile data and extract the attributes necessary for soil classification.  

scripts/ml_train_class_balance_weighted.py: Use this script to train the XGBoost modeel.

script/align_raster.py: Use this script to align all the predictors to the reference raster. Put all the predictors in 'path' and the 'reference' is the reference raster.

script/Raster_grids.py: Use this script to split the predictors into different smaller grids for ease of computation. Put all the input rsaters in 'pathRaster' (Line 24)

script/predict_model.py: Use this script to predict the model outputs across the nation.



View interactive Juxtapose map: [Before/After Map](https://kmu00009.github.io/SoilMapping/juxtapose.html)

