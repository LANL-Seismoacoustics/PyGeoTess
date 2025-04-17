"""
Test GeoTessGrid methods.

"""
from pathlib import Path

import numpy as np
import pytest

from geotess import lib

# datadir = Path(geotess.__file__).parents[0] / 'data'
# inputfile = str(datadir / 'geotess_grid_64000.geotess')
# grid_id = '90E53A213AEC248687F0D661B46D194A'

testdata = Path(__file__).parents[0] / 'testdata'

@pytest.fixture(scope="module")
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

def test_getNTriangles(grid_unified):
    grid = grid_unified['grid']
    data = (
        (20, (0,0)),
		(80, (0,1)),
		(320, (0,2)),

		(20, (1,0)),
		(80, (1,1)),
		(320, (1,2)),
		(1280, (1,3)),

		(20, (2,0)),
		(80, (2,1)),
		(320, (2,2)),
		(1280, (2,3)),
		(5120, (2,4)),
		(20480, (2,5)),
		(54716, (2,6)),
		(60224, (2,7)),
    )
    for expected, (tess, level) in data:
        assert grid.getNTriangles(tess, level) == expected

	# void testGetTriangleIntIntInt()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetTriangleIntIntInt" << endl;

	# 	TS_ASSERT_EQUALS(15, grid->getTriangle(0,0,15));
	# 	TS_ASSERT_EQUALS(35, grid->getTriangle(0,1,15));
	# 	TS_ASSERT_EQUALS(115, grid->getTriangle(0,2,15));
	# 	TS_ASSERT_EQUALS(435, grid->getTriangle(1,0,15));
	# 	TS_ASSERT_EQUALS(455, grid->getTriangle(1,1,15));
	# 	TS_ASSERT_EQUALS(535, grid->getTriangle(1,2,15));
	# 	TS_ASSERT_EQUALS(855, grid->getTriangle(1,3,15));
	# 	TS_ASSERT_EQUALS(2135, grid->getTriangle(2,0,15));
	# 	TS_ASSERT_EQUALS(2155, grid->getTriangle(2,1,15));
	# 	TS_ASSERT_EQUALS(2235, grid->getTriangle(2,2,15));
	# 	TS_ASSERT_EQUALS(2555, grid->getTriangle(2,3,15));
	# 	TS_ASSERT_EQUALS(3835, grid->getTriangle(2,4,15));
	# 	TS_ASSERT_EQUALS(8955, grid->getTriangle(2,5,15));
	# 	TS_ASSERT_EQUALS(29435, grid->getTriangle(2,6,15));
	# 	TS_ASSERT_EQUALS(84151, grid->getTriangle(2,7,15));
	# }

def test_getFirstTriangle(grid_unified):
    grid = grid_unified['grid']
    data = (
        (0, (0,0)),
        (20, (0,1)),
        (100, (0,2)),
        (420, (1,0)),
        (440, (1,1)),
        (520, (1,2)),
        (840, (1,3)),
        (2120, (2,0)),
        (2140, (2,1)),
        (2220, (2,2)),
        (2540, (2,3)),
        (3820, (2,4)),
        (8940, (2,5)),
        (29420, (2,6)),
        (84136, (2,7)),
    )
    for expected, (tess, level) in data:
        assert grid.getFirstTriangle(tess, level) == expected

def test_getLastTriangle(grid_unified):
    grid = grid_unified['grid']
    data = (
        (19, (0,0)),
		(99, (0,1)),
		(419, (0,2)),
		(439, (1,0)),
		(519, (1,1)),
		(839, (1,2)),
		(2119, (1,3)),
		(2139, (2,0)),
		(2219, (2,1)),
		(2539, (2,2)),
		(3819, (2,3)),
		(8939, (2,4)),
		(29419, (2,5)),
		(84135, (2,6)),
		(144359, (2,7)),
    )
    for expected, (tess, level) in data:
        assert grid.getLastTriangle(tess, level) == expected

	# void testGetTriangles()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetTriangles" << endl;

	# 	TS_ASSERT_EQUALS(232, grid->getTriangles()[1000][0]);;
	# }

	# void testGetEdges()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetEdges" << endl;

	# 	// if (model->getMetaData().getOptimizationType() == GeoTessOptimizationType::SPEED)
	# 	// {
	# 		double v[] = {-0.36180339887498947, 0.2628655560595669, -0.27639320225002106};
	# 		TS_ASSERT(Compare::arrays(grid->getEdgeList(65)[1]->normal, v, 1e-15));
	# 	// }
	# }

	# void testGetTriangleVertexIndex()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetTriangleVertexIndex" << endl;

	# 	TS_ASSERT_EQUALS(24, grid->getTriangleVertexIndex(65, 1));
	# }

	# void testGetTriangleVertex()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetTriangleVertex" << endl;

	# 	double v[] = {0.85065080835204, -6.525727206302101E-17, -0.5257311121191336};
	# 	TS_ASSERT(Compare::arrays(grid->getTriangleVertex(62, 1), v, 1e-15));
	# }

	# void testGetTriangleVertices()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetTriangleVertices" << endl;

	# 	double** actual = CPPUtils::new2DArray<double>(3,3);

	# 	grid->getTriangleVertices(333, actual);

	# 	double expected0[] = {0.723606797749979, -0.5257311121191337, -0.447213595499958};
	# 	TS_ASSERT(Compare::arrays(actual[0], expected0, 1e-15));

	# 	double expected1[] = {0.5127523743216502, -0.6937804775604494, -0.5057209226277919};
	# 	TS_ASSERT(Compare::arrays(actual[1], expected1, 1e-15));

	# 	double expected2[] = {0.6816403771773872, -0.6937804775604494, -0.23245439371512025};
	# 	TS_ASSERT(Compare::arrays(actual[2], expected2, 1e-15));

	# 	CPPUtils::delete2DArray(actual);

	# }

def test_getTriangleVertexIndexes(grid_unified):
    grid = grid_unified['grid']
    expected = np.array([1, 24, 23])
    np.testing.assert_equal(grid.getTriangleVertexIndexes(65), expected)

	# void testGetCircumCenter()
	# {
	# 	if (Compare::verbosity() > 0)
	# 		cout << "GeoTessGridTest::testGetCircumCenter" << endl;

	# 	double v[] = {0.6372374384402482, -0.662437103193734, -0.3938343958599925};

	# 	grid->computeCircumCenters();
	# 	const double* c = grid->getCircumCenter(333);
	# 	TS_ASSERT(Compare::arrays(c, v, 1e-15));
	# }

def test_testGrid(grid_unified):
    grid = grid_unified['grid']
    grid.testGrid()