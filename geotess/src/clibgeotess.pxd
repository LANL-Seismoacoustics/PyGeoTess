"""
This module pulls GeoTess c++ functionality into Cython (not Python yet).

We pull from all of GeoTess into this one place, and only the desired classes,
methods, and functions.  Declarations are unchanged from the original.

"""
from libcpp.string cimport string

cdef extern from "GeoTessGrid.h" namespace "geotess":
    cdef cppclass GeoTessGrid:
        GeoTessGrid() except +
        # string value inputFile is turned into a pointer, that can't be used to
        # modify the thing it points to, and returns a pointer to a GeoTessGrid.
        GeoTessGrid* loadGrid(const string& inputFile)
        int getNLevels()
        int getNTriangles()
        int getNTessellations()
        string toString()

