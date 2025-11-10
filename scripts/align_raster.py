# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:42:40 2025

@author: km000009
"""

from osgeo import gdal
import os
import glob


def align_raster(input_raster_path, reference_raster_path):
    # Open the input and reference rasters
    input_raster = gdal.Open(input_raster_path)
    reference_raster = gdal.Open(reference_raster_path)

    if input_raster is None or reference_raster is None:
        print(f"Error opening rasters: {input_raster_path} or {reference_raster_path}")
        return

    # Get reference geotransform and projection
    reference_geotransform = reference_raster.GetGeoTransform()
    reference_projection = reference_raster.GetProjection()
    reference_x_size = reference_raster.RasterXSize
    reference_y_size = reference_raster.RasterYSize

    # Create a temporary raster (same driver as input to avoid format issues)
    driver = gdal.GetDriverByName('GTiff')
    temp_raster_path = input_raster_path + "_temp.tif"
    temp_raster = driver.Create(
        temp_raster_path,
        reference_x_size,
        reference_y_size,
        input_raster.RasterCount,
        gdal.GDT_Float32
    )
    temp_raster.SetGeoTransform(reference_geotransform)
    temp_raster.SetProjection(reference_projection)

    # Set NoData value
    for i in range(1, input_raster.RasterCount + 1):
        band = temp_raster.GetRasterBand(i)
        band.SetNoDataValue(-9999)

    # Reproject and resample
    gdal.ReprojectImage(
        input_raster,
        temp_raster,
        input_raster.GetProjection(),
        reference_projection,
        gdal.GRA_NearestNeighbour
    )

    # Close datasets
    input_raster = None
    reference_raster = None
    temp_raster = None

    # Replace the input raster with the temporary aligned raster
    os.remove(input_raster_path)            # Delete the original file
    os.rename(temp_raster_path, input_raster_path)  # Rename temp to original name

    print(f"Input raster {input_raster_path} has been aligned and replaced.")
    
    
path = '../../../Data/predictData/cofactors_used'
reference = '../../../Data/predictData/Elevation_dm.tif'

rasters = glob.glob(os.path.join(path, '*.tif'))

# print(rasters)

for raster in rasters:
    align_raster(raster, reference)