"""
Test GeoTessGrid methods.

"""
from pathlib import Path

import numpy as np
import pytest

import geotess
from geotess import lib

# datadir = Path(geotess.__file__).parents[0] / 'data'
# inputfile = str(datadir / 'geotess_grid_64000.geotess')
# grid_id = '90E53A213AEC248687F0D661B46D194A'

testdata = Path(__file__).parents[0] / 'testdata'

@pytest.fixture
def grid_unified():
    inputfile = str(testdata / 'unified_crust20_ak135_grid.geotess')
    grid = lib.GeoTessGrid()
    grid.loadGrid(inputfile)
    grid_id = "B0539B9CC5512D2D625A7593B74BE4A7"

    out = {
        'grid': grid,
        'inputfile': inputfile,
        'grid_id': grid_id,
        }

    return out


@pytest.fixture
def grid16():
    inputfile = str(testdata / 'geotess_grid_16000.geotess')
    grid = lib.GeoTessGrid()
    grid.loadGrid(inputfile)
    grid_id = "4FD3D72E55EFA8E13CA096B4C8795F03" 

    out = {
        'grid': grid,
        'inputfile': inputfile,
        'grid_id': grid_id,
        }

    return out


def test_init():
    # Test empty/nullary constructor
    # This just tests that it doesn't crash the interpreter.
    g = lib.GeoTessGrid()
    del g 

def test_loadGrid(grid16):
    expected = grid16
    grid = lib.GeoTessGrid()
    grid.loadGrid(expected['inputfile'])
    assert grid.getGridID() == expected['grid_id']

def test_getGridInputFile(grid16):
    expected = grid16
    grid = expected['grid']
    assert grid.getGridInputFile() == expected['inputfile']

def test_getGridID(grid16):
    expected = grid16
    grid = expected['grid']
    assert grid.getGridID() == expected['grid_id']

def test_getNVertices(grid_unified):
    grid = grid_unified['grid']
    expected = 30114
    assert grid.getNVertices() == expected

def test_getNTessellations(grid_unified):
    grid = grid_unified['grid']
    expected = 3
    assert grid.getNTessellations() == expected

def test_getNLevels(grid_unified):
    grid = grid_unified['grid']
    assert grid.getNLevels(0) == 3
    assert grid.getNLevels(1) == 4
    assert grid.getNLevels(2) == 8
    assert grid.getNLevels() == 15

	# void testGetLevel()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetLevel" << endl;

	# 	TS_ASSERT_EQUALS(2, grid->getLevel(0, 2));
	# 	TS_ASSERT_EQUALS(5, grid->getLevel(1, 2));
	# 	TS_ASSERT_EQUALS(9, grid->getLevel(2, 2));
	# }

	# void testGetLastLevel()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetLastLevel" << endl;

	# 	TS_ASSERT_EQUALS(2, grid->getLastLevel(0));
	# 	TS_ASSERT_EQUALS(6, grid->getLastLevel(1));
	# 	TS_ASSERT_EQUALS(14, grid->getLastLevel(2));
	# }

	# void testGetGridSoftwareVersion()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetGridSoftwareVersion" << endl;

	# 	TS_ASSERT_EQUALS("GeoModel 7.0.1", grid->getGridSoftwareVersion());
	# }

	# void testGetGridGenerationDate()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetGridGenerationDate" << endl;

	# 	TS_ASSERT_EQUALS("Wed April 18 15:21:51 2012",
	# 			grid->getGridGenerationDate());
	# }


def test_getVertex(grid_unified):
    grid = grid_unified['grid']
    expected = np.array([0.36180339887498947, 0.26286555605956685, 0.8944271909999159])
    vertex = grid.getVertex(42)
    np.testing.assert_allclose(vertex, expected, atol=1e-15)
    
    # 	void testGetVertexIntIntIntInt()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetVertexIntIntIntInt" << endl;

	# 	TS_ASSERT(Compare::arrays(grid->getVertex(0,0,10,2), 1e-15, 3, 0.7236067977499789, -0.5257311121191336, -0.4472135954999579));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(0,1,10,2), 1e-15, 3, -0.4253254041760201, -0.3090169943749476, 0.8506508083520398));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(0,2,10,2), 1e-15, 3, 0.5013752464907345, 0.702046444776163, 0.5057209226277919));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(1,0,10,2), 1e-15, 3, 0.7236067977499789, -0.5257311121191336, -0.4472135954999579));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(1,1,10,2), 1e-15, 3, -0.4253254041760201, -0.3090169943749476, 0.8506508083520398));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(1,2,10,2), 1e-15, 3, 0.5013752464907345, 0.702046444776163, 0.5057209226277919));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(1,3,10,2), 1e-15, 3, 0.44929887015742925, 0.13307110414059134, 0.8834153080618772));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,0,10,2), 1e-15, 3, 0.7236067977499789, -0.5257311121191336, -0.4472135954999579));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,1,10,2), 1e-15, 3, -0.4253254041760201, -0.3090169943749476, 0.8506508083520398));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,2,10,2), 1e-15, 3, 0.5013752464907345, 0.702046444776163, 0.5057209226277919));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,3,10,2), 1e-15, 3, 0.44929887015742925, 0.13307110414059134, 0.8834153080618772));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,4,10,2), 1e-15, 3, 0.5002770524523549, 0.4642134014223056, 0.7309095626201076));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,5,10,2), 1e-15, 3, 0.4837287583319597, 0.30052751433074043, 0.8220034680540023));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,6,10,2), 1e-15, 3, 0.49407785976537, 0.3845000537527731, 0.7797735422247833));
	# 	TS_ASSERT(Compare::arrays(grid->getVertex(2,7,10,2), 1e-15, 3, -0.2129707343686073, -0.6387308869226458, 0.739368866259262));
	# }

def test_getVertexIndex(grid_unified):
    grid = grid_unified['grid']
    data = (
        (10, (10,2,0,0)),
        (18, (10,2,0,1)),
        (48, (10,2,0,2)),
        (10, (10,2,1,0)),
        (18, (10,2,1,1)),
        (48, (10,2,1,2)),
        (168, (10,2,1,3)),
        (10, (10,2,2,0)),
        (18, (10,2,2,1)),
        (48, (10,2,2,2)),
        (168, (10,2,2,3)),
        (648, (10,2,2,4)),
        (2568, (10,2,2,5)),
        (10248, (10,2,2,6)),
        (27367, (10,2,2,7)),
    )
    for expected, indices in data:
        assert grid.getVertexIndex(*indices) == expected

def test_getVertices(grid_unified):
    grid = grid_unified['grid']
    expected = np.array([0.5002770524523549, 0.4642134014223056, 0.7309095626201076])
    vertices = grid.getVertices()
    np.testing.assert_allclose(vertices[648], expected, atol=1e-15)