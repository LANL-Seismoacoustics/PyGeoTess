import os

from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

np.import_array()

cimport geotess.lib.geotesscpp as clib
import geotess.exc as exc


cdef class GeoTessGrid:
    cdef clib.GeoTessGrid *thisptr
    cdef object owner

    def __cinit__(self, raw=False):
        # XXX: lots of things evaluate to True or False. A file name, for example.
        if not raw:
            self.thisptr = new clib.GeoTessGrid()

    def __dealloc__(self):
        if self.thisptr != NULL and not self.owner:
            del self.thisptr

    def loadGrid(self, const string& inputFile):
        if os.path.exists(inputFile):
            self.thisptr.loadGrid(inputFile)
        else:
            raise exc.GeoTessFileError("File not found.")

    def writeGrid(self, const string& fileName):
        self.thisptr.writeGrid(fileName)

    def getNLevels(self):
        return self.thisptr.getNLevels()
 
    def getNTriangles(self, tessellation=None, level=None):
        if tessellation is None and level is None:
            NTriangles = self.thisptr.getNTriangles()
        else:
            Nlevels = self.thisptr.getNLevels() 
            NTess = self.getNTessellations()
            if level > Nlevels or tessellation > NTess:
                msg = "level > {} or tessellation > {}".format(Nlevels, NTess)
                raise ValueError(msg)
            NTriangles = self.thisptr.getNTriangles(int(tessellation), int(level))

        return NTriangles

    def getNTessellations(self):
        return self.thisptr.getNTessellations()

    def getNVertices(self):
        return self.thisptr.getNVertices()

    def getVertices(self):
        # http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html#create-cython-wrapper-class
        # _grid.vertices() returns a double const* const* (2D array), which will
        # need some internal NumPy functions to get out of Cython.
        # http://stackoverflow.com/questions/27940848/passing-2-dimensional-c-array-to-python-numpy
        cdef int nVert = self.thisptr.getNVertices()
        cdef int nCol = 3
        cdef np.npy_intp Dims[2]
        Dims[0] = nVert
        Dims[1] = nCol

        # an nVert X 3 2D array
        cdef const double *const * c_vertices
        c_vertices = self.thisptr.getVertices()

        # http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#c.PyArray_SimpleNew
        # PyObject *PyArray_SimpleNew(int nd, npy_intp* dims, int typenum)
        # Allocate the memory needed for the array
        cdef np.ndarray ArgsArray = np.PyArray_SimpleNew(2, Dims, np.NPY_DOUBLE)
        # The pointer to the array data is accessed using PyArray_DATA()
        cdef double *p = <double *> np.PyArray_DATA(ArgsArray)

        for r in range(nVert):
            memcpy(p, c_vertices[r], sizeof(double) * nCol)
            p += nCol

        return ArgsArray

    def toString(self):
        return self.thisptr.toString()

    def getVertex(self, int vertex):
        """
        Retrieve the unit vector that corresponds to the specified vertex.

        Returns a 3-element NumPy vector.  This array is still connected to the
        vertex in-memory, so don't modify it unless you intend to!

        """
        # Use some internal NumPy C API calls to safely wrap the array pointer,
        # hopefully preventing memory leaks or segfaults.
        # following https://gist.github.com/aeberspaecher/1253698
        cdef const double *vtx = self.thisptr.getVertex(vertex)

        # Use the PyArray_SimpleNewFromData function from numpy to create a
        # new Python object pointing to the existing data
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 3
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void *> vtx)

        # Tell Python that it can deallocate the memory when the ndarray
        # object gets garbage collected
        # As the OWNDATA flag of an array is read-only in Python, we need to
        # call the C function PyArray_UpdateFlags
        np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_OWNDATA)
        # http://stackoverflow.com/questions/19204098/c-code-within-python-and-copying-arrays-in-c-code

        # XXX: this seems to contradict the docstring that memory is shared.
        # I must've done it just to be safe, even though it doesn't follow the 
        # original API.
        return arr.copy()

    def getVertexTriangles(self, int tessId, int level, int vertex):
        """
        Retrieve a list of the triangles a particular vertex is a member of,
        considering only triangles in the specified tessellation/level.

        """
        # getVertexTriangles(const int &tessId, const int &level, const int &vertex) const
        # This calling signature removes the "const" from the C++ method signature.
        # XXX: check that self has enough tessellations, levels, vertices to
        #   return what you requested
        # XXX: also check that inputs are indeed integers

        cdef vector[int] triangles = self.thisptr.getVertexTriangles(tessId, level, vertex)
        # automatic conversion from vector[int] to Python list:
        # https://github.com/cython/cython/blob/master/tests/run/cpp_stl_conversion.pyx

        return triangles

    def getTriangleVertexIndexes(self, int triangleIndex):
        """
        Supply an integer triangle index, get a 3-element integer array, which
        are indices of the vertices that make this triangle.

        """
        cdef const int *tri_vertex_ids = self.thisptr.getTriangleVertexIndexes(triangleIndex)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 3
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> tri_vertex_ids)
        np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_OWNDATA)

        return arr.copy()

    def getFirstTriangle(self, int tessellation, int level):
        return self.thisptr.getFirstTriangle(tessellation, level)

    def getLastTriangle(self, int tessellation, int level):
        return self.thisptr.getLastTriangle(tessellation, level)

    def getVertexIndex(self, int triangle, int corner):
        return self.thisptr.getVertexIndex(triangle, corner)

    @staticmethod
    cdef GeoTessGrid wrap(clib.GeoTessGrid *cptr, owner=None):
        # This is a Cython helper function that facilitates passing ownership
        # of a C++ pointer to a Python class
        # XXX: I don't think this is working
        cdef GeoTessGrid inst = GeoTessGrid(raw=True)
        inst.thisptr = cptr
        if owner:
            inst.owner = owner

        return inst



