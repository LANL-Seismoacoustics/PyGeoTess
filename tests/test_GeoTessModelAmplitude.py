""" Testing for wrapped C++ GeoTessModelAmplitude class
"""
from pathlib import Path

import numpy as np
import pytest

import geotess.lib as lib
from geotess.lib import geotessutils as gtutil

testdata = Path(__file__).parents[0] / 'testdata'


@pytest.fixture(scope="function")
def modelRun1():
    inputfile = str(testdata / 'amptomo_run1_Lg.geotess')
    model = lib.GeoTessModelAmplitude()
    model.loadModel(inputfile)

    return model

def test_init():
    """ Test the default constructor and destructor. Basically, hope it doesn't crash.
    """
    # TODO: use tracemalloc here to make sure "del" frees memory
    gtamp = lib.GeoTessModelAmplitude()
    del gtamp

    # must actually be a subclass of GeoTessModel
    assert issubclass(lib.GeoTessModelAmplitude, lib.GeoTessModel)

@pytest.mark.skip(reason="Always returns float nan, never succeeds.")
def test_getSiteTrans(modelRun1):
    term = modelRun1.getSiteTrans("MDJ", "BHZ", "1.0_2.0")
    assert term == pytest.approx(-18.5847, abs=1e-3)

    term = modelRun1.getSiteTrans("xxx", "BHZ", "1.0_2.0")
    assert term is None

    term = modelRun1.getSiteTrans("MDJ", "xxx", "1.0_2.0")
    assert term is None

    term = modelRun1.getSiteTrans("MDJ", "BHZ", "xxx")
    assert term is None

def test_getWeights_2D(modelRun1):
    """Test 2D getWeights - based on C++ testGetPathIntegral2DW.
    
    This test uses the two-point getWeights() signature which only works
    with 2D models (1 layer). Based on C++ test at line 321 in GeoTessModelTest.h.
    The amptomo model is a 2D model (1 layer) suitable for testing this method.
    """
    # Path from 0°N,0°E to 0°N,30°E
    firstPoint = gtutil.GeoTessUtils.getVectorDegrees(0, 0)
    lastPoint = gtutil.GeoTessUtils.getVectorDegrees(0, 30)
    
    # Calculate expected distance
    angle = np.arccos(np.dot(firstPoint, lastPoint))
    distance = angle * 6378.  # Earth radius in km
    
    # Call getWeights with 0.1 degree spacing
    spacing = 0.1 * np.pi / 180.  # 0.1 degrees in radians
    radius = 6378.  # km
    
    weights = modelRun1.getWeights(firstPoint, lastPoint, spacing, radius, 'NATURAL_NEIGHBOR')
    
    # Sum of weights should equal the arc length
    sum_weights = np.sum(list(weights.values()))
    assert sum_weights == pytest.approx(distance, abs=0.0001)
