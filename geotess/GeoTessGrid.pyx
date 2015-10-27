"""
GeoTessGrid Cython definitions.

The Cython definitions here are meant to expose direct functionality from the
underlying C++ classes.  Not all methods/constructors are exposed, only the
ones that are used in the pure Python wrapper class.  For example, the Python
Grid class has a constructor just uses the c++ loadGrid method, not any actual
c++ GeoTessGrid constructor, so no c++ constructor is exposed.

"""
# distutils: language = c++
# distutils: sources = GeoTessGrid.cc

cdef extern from "GeoTessGrid.h" namespace "geotess":
    GeoTessGrid() except +
    int getNLevels()
    int getNTriangles()
    int getNTessellations()
