""" Test low-level Cython wrapping of GeoTessMetaData C++ class.
"""
import geotess.lib as libgt

def test_init():
    # Test empty/nullary constructor
    # Just don't want it to crash the interpreter.
    md = libgt.GeoTessMetaData()
    del md
