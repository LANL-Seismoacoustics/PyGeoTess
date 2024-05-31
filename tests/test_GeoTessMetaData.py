""" Test low-level Cython wrapping of GeoTessMetaData C++ class.
"""
import geotess.libgeotess as libgt

class TestGeoTessMetaData:
    def test_init(self):
        # Test empty/nullary constructor
        # Just don't want it to crash the interpreter.
        md = libgt.GeoTessMetaData()
        del md

# def test_GeoTessMetaData__init__():
#     # Test empty/nullary constructor
#     # Just don't want it to crash the interpreter.
#     md = libgt.GeoTessMetaData()
#     del md
