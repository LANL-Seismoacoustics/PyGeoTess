""" Testing for wrapped C++ GeoTessModelAmplitude class
"""
import geotess.libgeotess as libgt


def test_init():
    """ Test the default constructor and destructor. Basically, hope it doesn't crash.
    """
    # TODO: use tracemalloc here to make sure "del" frees memory
    gtamp = libgt.GeoTessModelAmplitude()
    del gtamp

    # must actually be a subclass of GeoTessModel
    assert issubclass(libgt.GeoTessModelAmplitude, libgt.GeoTessModel)