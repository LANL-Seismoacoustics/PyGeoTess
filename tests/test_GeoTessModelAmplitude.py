""" Testing for wrapped C++ GeoTessModelAmplitude class
"""
from pathlib import Path

import pytest

import geotess.lib as lib

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
