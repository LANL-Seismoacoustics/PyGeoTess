""" Tests for libgeotess.GeoTessUtils
"""
import numpy as np
import geotess.libgeotess as libgt
import pytest

LAT = 30.0
LON = 45.0
R = 6372.824420335703
VEC = np.array([0.61339643, 0.61339643, 0.4974833])

def test_init():
    utils = libgt.GeoTessUtils()
    del utils

def test_getLatDegrees():
    expected = LAT
    observed = libgt.GeoTessUtils.getLatDegrees(VEC)
    np.testing.assert_approx_equal(expected, observed)

def test_getLonDegrees():
    expected = LON
    observed = libgt.GeoTessUtils.getLonDegrees(VEC)
    np.testing.assert_approx_equal(expected, observed)

def test_getVectorDegrees():
    expected = VEC
    observed = libgt.GeoTessUtils.getVectorDegrees(LAT, LON)
    np.testing.assert_array_almost_equal(expected, observed)

def test_getEarthRadius():
    expected = R
    observed = libgt.GeoTessUtils.getEarthRadius(VEC)
    assert observed == pytest.approx(expected)
