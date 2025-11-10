import os
import time
import glob
from predictFunctions import extractExtent, splitShape, ReadGlaExtent, ReadGlaPoly


# Main execution
start = time.time()
print('Extract extents from the grids...')
path = "../../../Data"
grid = os.path.join(path, "England_ALC50_GRID.shp")
extractExtent(grid, path)

print('Prepare shapefile for each grid....')
grid = os.path.join("../../../Data/England_ALC50_GRID.shp")
outgridpath = '../../../Data/GridsPoly'
os.makedirs(outgridpath, exist_ok=True)
# split the shape file in different polygons
splitShape(grid, outgridpath)


print('Split the input rasters into grids...')
# path to the tif file which needs to be split
pathRaster = '../../../Data/predictData/cofactors_used'

# path to the shapefiles extents
gridPath = '../../../Data/ExtPath.txt'
extents = '../../../Data/Extent.txt'

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
    
