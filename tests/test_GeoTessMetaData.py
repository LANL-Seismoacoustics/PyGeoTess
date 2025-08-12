""" Test low-level Cython wrapping of GeoTessMetaData C++ class.
"""
# import filecmp
from pathlib import Path
# from tempfile import NamedTemporaryFile

import array
import numpy as np
import pytest

from geotess import lib

testdata = Path(__file__).parents[0] / 'testdata'

@pytest.fixture(scope="module")
def unified():
    inputfile = str(testdata / 'unified_crust20_ak135.geotess')
    model = lib.GeoTessModel()
    model.loadModel(inputfile)
    meta = model.getMetaData()

    return meta, inputfile


@pytest.fixture(scope="function")
def crust20():
    inputfile = str(testdata / 'crust20.geotess')
    model = lib.GeoTessModel()
    model.loadModel(inputfile)
    meta = model.getMetaData()

    return meta, inputfile


def test_init():
    # Test empty/nullary constructor
    # Just don't want it to crash the interpreter.
    md = lib.GeoTessMetaData()
    del md


def test_getDescription(crust20):
    meta, _ = crust20
    expected = 'crust 2.0\nLaske and Masters\n'
    observed = meta.getDescription()
    assert observed == expected

def test_setDescription(crust20):
    meta, _ = crust20
    expected = 'an awesome\nmodel\n'
    meta.setDescription(expected)
    observed = meta.getDescription()
    assert observed == expected

def test_setLayerNames(crust20):
    meta, _ = crust20
    expected = "layer1;layer2;layer3;layer4;layer5;layer6;layer7"
    meta.setLayerNames(expected)
    observed = meta.getLayerNamesString()
    assert observed == expected

def test_getLayerNames(crust20):
    meta, _ = crust20
    expected = "mantle;lower_crust;middle_crust;upper_crust;hard_sediments;soft_sediments;ice"
    observed = meta.getLayerNamesString()
    assert observed == expected

def test_setLayerTessIds():
    meta = lib.GeoTessMetaData()
    expected = [0, 0, 1]
    # check types that support the buffer protocol
    inputs = [
        expected,
        tuple(expected),
        np.array(expected),
        array.array('B', expected),
    ]
    for inpt in inputs:
        meta.setLayerTessIds(inpt)
        observed = meta.getLayerTessIds()
        assert observed == expected
        assert meta.getNLayers() == 3


def test_getNLayers(crust20):
    meta, _ = crust20
    assert meta.getNLayers() == 7


def test_getLayerName(crust20):
    meta, _ = crust20
    layer_names = [
        "mantle",
        "lower_crust",
        "middle_crust",
        "upper_crust",
        "hard_sediments",
        "soft_sediments",
        "ice",
    ]
    for index, expected in enumerate(layer_names):
        assert meta.getLayerName(index) == expected

def test_getLayerIndex(crust20):
    meta, _ = crust20
    layer_names = [
        "mantle",
        "lower_crust",
        "middle_crust",
        "upper_crust",
        "hard_sediments",
        "soft_sediments",
        "ice",
    ]
    for expected, name in enumerate(layer_names):
        assert meta.getLayerIndex(name) == expected
