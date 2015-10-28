"""
This module exposes Cython GeoTess functionality from the pxd file into Python.

Using both a pxd and a pyx file is done, partly, so that we can keep the
exposed c++ GeoTess functionality together in one namespace using "cimport",
and we can name the classes exposed to Python the same as those in the
GeoTess c++.

The Python-visible definitions here are just wrapper classes that map directly
to their Cython-exposed c++ counterparts in the corresponding pxd file.

"""
cimport clibgeotess as clib

cdef class Grid:
    cdef GeoTessGrid *thisptr
    def _cinit__(self):
        self.thisptr = new GeoTessGrid
    def __dealloc__(self):
        del self.thisptr
    def getNLevels(self):
        return self.thisptr.getNLevels()
    def getNTriangles(self):
        return self.thisptr.getNTriangles()
    def getNTessellations(self):
        return self.thisptr.getNTessellations()
