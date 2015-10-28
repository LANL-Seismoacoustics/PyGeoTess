"""
This module pulls GeoTess c++ functionality into Cython (not Python yet).

We pull from all of GeoTess into this one place, and only the desired classes,
methods, and functions.  Declarations are unchanged from the original.

"""
cdef extern from "GeoTessGrid.h" namespace "geotess":
    cdef cppclass GeoTessGrid:
        GeoTessGrid() except +
        int getNLevels()
        int getNTriangles()
        int getNTessellations()

