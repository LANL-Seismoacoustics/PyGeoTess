"""
Test GeoTessGrid methods.

"""
from pathlib import Path

import pytest

import geotess
from geotess import lib

datadir = Path(geotess.__file__).parents[0] / 'data'
inputfile = str(datadir / 'geotess_grid_64000.geotess')
grid_id = '90E53A213AEC248687F0D661B46D194A'


@pytest.fixture
def grid_object():
    grid = lib.GeoTessGrid()
    grid.loadGrid(inputfile)

    return grid


def test_init():
    # Test empty/nullary constructor
    # Just Tests that it doesn't crash the interpreter.
    g = lib.GeoTessGrid()
    del g 

def test_loadGrid():
    grid = lib.GeoTessGrid()
    grid.loadGrid(inputfile)
    # TODO: use the getGridID instead
    # assert grid.getGridInputFile() == inputfile
    assert grid.getGridID() == grid_id

def test_getGridInputFile(grid_object):
    grid = grid_object
    assert grid.getGridInputFile() == inputfile

def test_getGridID(grid_object):
    grid = grid_object
    assert grid.getGridID() == grid_id
