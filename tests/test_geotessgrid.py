"""
Test GeoTessGrid methods.

"""
import os

import geotess
from geotess.lib.libgeotess import GeoTessGrid

# <install location>/geotess/data
# datadir = os.path.dirname(geotess.__file__) + os.path.sep + 'data'
datadir = '/Users/jkmacc2/code/PyGeoTess/GeoTessModels'


def test_init():
    # Test empty/nullary constructor
    # Just Tests that it doesn't crash the interpreter.
    g = GeoTessGrid()
    del g 

def test_loadGrid():
    inputfile = datadir + os.path.sep + 'geotess_grid_64000.geotess'
    grid = GeoTessGrid()
    grid.loadGrid(inputfile)
    assert grid.getGridInputFile() == inputfile
