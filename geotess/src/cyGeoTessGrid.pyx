"""
GeoTessGrid Cython definitions.

The Cython definitions here are meant to expose direct functionality from the
underlying C++ classes.  Not all methods/constructors are exposed, only the
ones that are used in the pure Python wrapper class.  For example, the Python
Grid class has a constructor just uses the c++ loadGrid method, not any actual
c++ GeoTessGrid constructor, so no c++ constructor is exposed.

"""
# distutils: language = c++

# this cdef extern block exposes the interface of the c++ libary
cdef extern from "GeoTessGrid.h" namespace "geotess":
    cdef cppclass GeoTessGrid:
        GeoTessGrid() except +
        int getNLevels()
        int getNTriangles()
        int getNTessellations()


# this cdef class block exposes a corresponding class to Python
cdef class Grid:
    cdef GeoTessGrid *thisptr
    def _cinit__(self):
        self.thisptr = new GeoTessGrid
