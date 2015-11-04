from datetime import datetime
import geotess.libgeotess as lib

md =  lib.GeoTessMetaData()
md.setEarthShape("WGS84")

descr = """
Simple example of populating a 3D GeoTess model
comprised of 3 multi-level tessellations
author: Sandy Ballard
contact: sballar@sandia.gov"""

md.setDescription(descr)

md.setLayerNames("INNER_CORE; OUTER_CORE; LOWER_MANTLE; TRANSITION_ZONE; UPPER_MANTLE; LOWER_CRUST; UPPER_CRUST")
md.setLayerTessIds([0, 0, 1, 1, 1, 2, 2])
md.setAttributes("Vp; Vs; rho", "km/sec; km/sec; g/cc")
md.setDataType('FLOAT')
md.setModelSoftwareVersion("GeoTessCPPExamples.PopulateModel3D 1.0.0")
md.setModelGenerationDate(str(datetime.now()))

grid = lib.GeoTessGrid.loadGrid('../data/small_model_grid.ascii')

# Initialize the model.  It is full of null data.
model = lib.GeoTessModel(grid, md)

# a helpful human-readable form of model info
print(model.toString())
