"""
This module pulls GeoTess c++ functionality into Cython (not Python yet).

We pull from all of GeoTess into this one place, and only the desired classes,
methods, and functions.  Declarations are unchanged from the original.

"""
# distutils: language = c++

# this cdef extern block exposes the interface of the c++ libary to Cython
# GeoTessGrid is still not visible from Python yet.
cdef extern from "GeoTessGrid.h" namespace "geotess":
    cdef cppclass GeoTessGrid:
        GeoTessGrid() except +
        int getNLevels()
        int getNTriangles()
        int getNTessellations()

