""" Tests for libgeotess.GeoTessUtils
"""
import numpy as np
import geotess.libgeotess as lib

LAT = 30.0
LON = 45.0
VEC = np.array([0.61339643, 0.61339643, 0.4974833])

def test_init():
    utils = lib.GeoTessUtils()
    del utils

def test_getLatDegrees():
    expected = LAT
    observed = lib.GeoTessUtils.getLatDegrees(VEC)
    np.testing.assert_approx_equal(expected, observed)

def test_getLonDegrees():
    expected = LON
    observed = lib.GeoTessUtils.getLonDegrees(VEC)
    np.testing.assert_approx_equal(expected, observed)

def test_getVectorDegrees():
    expected = VEC
    observed = lib.GeoTessUtils.getVectorDegrees(LAT, LON)
    np.testing.assert_array_almost_equal(expected, observed)