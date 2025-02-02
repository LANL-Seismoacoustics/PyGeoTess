"""
Test GeoTessGrid methods.

"""
import os

import geotess
from geotess.lib.libgeotess import GeoTessGrid

# <install location>/geotess/data
# datadir = os.path.dirname(geotess.__file__) + os.path.sep + 'data'
datadir = '/Users/jkmacc2/code/PyGeoTess/GeoTessModels'
inputfile = datadir + os.path.sep + 'geotess_grid_64000.geotess'

grid = GeoTessGrid()
grid.loadGrid(inputfile)
grid_id = '90E53A213AEC248687F0D661B46D194A'

def test_init():
    # Test empty/nullary constructor
    # Just Tests that it doesn't crash the interpreter.
    g = GeoTessGrid()
    del g 

def test_loadGrid():
    inputfile = datadir + os.path.sep + 'geotess_grid_64000.geotess'
    grid = GeoTessGrid()
    grid.loadGrid(inputfile)
    # TODO: use the getGridID instead
    assert grid.getGridInputFile() == inputfile

def test_getGridID():
    assert grid.getGridID() == grid_id