#distutils: language = c++
#cython: embedsignature=True
"""
This module exposes Cython GeoTess functionality from the pxd file into Python.

The class definitions here are Python-visible, and are simply wrappers that 
forward the Python-exposed methods directly down to their Cython-exposed C++
counterparts, which have been exposed in the imported pxd file.

This module is also responsible for converting between Python types and c++
types, which sometimes involves annoying tricks.  For simple numerical types,
this conversion can be done automatically in the calling signature of a "def"
method if types are declared.  Complex c++ class types, for example, can't be
in a Python-visible "def" method because Python objects can't be automatically
cast to c++ types.  For these cases, sneaky factory functions that can accept
the complex types must do the work.  Unfortunately, this means that any
constructor or method that accepts complex c++ can't be "directly" exposed to
Python.

Using both a pxd and a pyx file is done, partly, so that we can keep the
exposed c++ GeoTess functionality together in one namespace using "cimport",
such that we can name the classes exposed to Python the same as those in the
GeoTess c++.  This is sometimes confusing in error messages, however.

GeoTess functionality is intentionally a one-to-one translation into Python
here so that any modifications to the way models and grids are used can be
developed and tested in other pure-Python modules.  This makes it easier to try
different Pythonic approaches to working with the underlying GeoTess library.


## Current conversion conventions

* NumPy vectors are generally used instead of lists or vectors, such as for
  GeoTess unit vectors and profiles.

* If a C++ method accepts an empty array/vector argument to be filled by
  the method, I leave that out of the calling signature.  It is instead
  initialized inside the method and simply returned by it.


## Current headaches

* Deleting or garbage-collecting objects is dangerous.  Some objects are
  managed by other objects, so deleting them manually can crash the interpreter.
  I'm not sure how to fix this yet.
 
* There is very little/no type checking between Python arguments and when
  they're forwarded to the c++ methods.  This is dangerous.

## Original C++ documentation
http://www.sandia.gov/geotess/assets/documents/documentation_cpp/annotated.html

"""
import os

# from cpython cimport array
# import array

import numpy as np
cimport numpy as np

np.import_array()

from cython.operator cimport dereference as deref

from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport clibgeotess as clib
import geotess.exc as exc


cdef class GeoTessUtils:
    cdef clib.GeoTessUtils *thisptr

    def __cinit__(self):
        self.thisptr = new clib.GeoTessUtils()

    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr

    @staticmethod
    def getLatDegrees(double[:] v):
        return clib.GeoTessUtils.getLatDegrees(&v[0])

    @staticmethod
    def getLonDegrees(double[:] v):
        return clib.GeoTessUtils.getLonDegrees(&v[0])


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

        return arr.copy()

    def getVertexTriangles(self, int tessId, int level, int vertex):
        """
        Retrieve a list of the triangles a particular vertex is a member of,
        considering only triangles in the specified tessellation/level.

        """
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
        cdef GeoTessGrid inst = GeoTessGrid(raw=True)
        inst.thisptr = cptr
        if owner:
            inst.owner = owner

        return inst


cdef class GeoTessMetaData:
    cdef clib.GeoTessMetaData *thisptr
    cdef object owner

    def __cinit__(self, raw=False):
        if not raw:
            self.thisptr = new clib.GeoTessMetaData()

    def __dealloc__(self):
        if self.thisptr != NULL and not self.owner:
            del self.thisptr

    def setEarthShape(self, str earthShapeName):
        shapes = ('SPHERE', 'WGS84', 'WGS84_RCONST', 'GRS80', 'GRS80_RCONST',
                  'IERS2003', 'IERS2003_RCONST')
        if earthShapeName not in shapes:
            msg = "Unknown earth shape '{}'. Choose from {}"
            raise ValueError(msg.format(earthShapeName, shapes))
        self.thisptr.setEarthShape(earthShapeName)

    def setDescription(self, const string& dscr):
        self.thisptr.setDescription(dscr)

    def setLayerNames(self, const string& lyrNms):
        self.thisptr.setLayerNames(lyrNms)

    def setLayerTessIds(self, vector[int]& layrTsIds):
        """
        layrTsIds is an iterable of integers.
        """
        # iterable of integers is automatically casted to vector of integers
        # http://www.peterbeerli.com/classes/images/f/f7/Isc4304cpluspluscython.pdf
        self.thisptr.setLayerTessIds(layrTsIds)

    def setAttributes(self, const string& nms, const string& unts):
        self.thisptr.setAttributes(nms, unts)

    def setDataType(self, dt):
        dtypes = ('DOUBLE', 'FLOAT', 'LONG', 'INT', 'SHORTINT', 'BYTE')
        if dt not in dtypes:
            raise ValueError("DataType must be one of {}".format(dtypes))
        self.thisptr.setDataType(dt)

    def setModelSoftwareVersion(self, const string& swVersion):
        self.thisptr.setModelSoftwareVersion(swVersion)

    def setModelGenerationDate(self, const string& genDate):
        self.thisptr.setModelGenerationDate(genDate)

    def toString(self):

        return self.thisptr.toString()

    @staticmethod
    cdef GeoTessMetaData wrap(clib.GeoTessMetaData *cptr, owner=None):
        cdef GeoTessMetaData inst = GeoTessMetaData(raw=True)
        inst.thisptr = cptr
        if owner:
            inst.owner = owner

        return inst

    def getAttributeNamesString(self):

        return self.thisptr.getAttributeNamesString()

    def getAttributeUnitsString(self):

        return self.thisptr.getAttributeUnitsString()

    def getLayerNamesString(self):

        return self.thisptr.getLayerNamesString()

    def getLayerTessIds(self):
        # Use some internal NumPy C API calls to safely wrap the array pointer,
        # hopefully preventing memory leaks or segfaults.
        # following https://gist.github.com/aeberspaecher/1253698
        cdef const int *tess_ids = self.thisptr.getLayerTessIds()
        cdef np.npy_intp shape[1]
        cdef int nLayers = self.thisptr.getNLayers()
        shape[0] = <np.npy_intp> nLayers
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> tess_ids)
        np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_OWNDATA)

        return arr.tolist()


cdef class EarthShape:
    """
    Parameters
    ----------
    earthShape : str
        SPHERE - Geocentric and geographic latitudes are identical and
            conversion between depth and radius assume the Earth is a sphere
            with constant radius of 6371 km.
        GRS80 - Conversion between geographic and geocentric latitudes, and
            between depth and radius are performed using the parameters of the
            GRS80 ellipsoid.
        GRS80_RCONST - Conversion between geographic and geocentric latitudes
            are performed using the parameters of the GRS80 ellipsoid.
            Conversions between depth and radius assume the Earth is a sphere
            with radius 6371.
        WGS84 - Conversion between geographic and geocentric latitudes, and
            between depth and radius are performed using the parameters of the
            WGS84 ellipsoid.
        WGS84_RCONST - Conversion between geographic and geocentric latitudes
            are performed using the parameters of the WGS84 ellipsoid.
            Conversions between depth and radius assume the Earth is a sphere
            with radius 6371.

    """
    cdef clib.EarthShape *thisptr
    cdef object owner

    def __cinit__(self, earthShape="WGS84", raw=False):
        # raw=True means "just give me the Python wrapper class, I don't want
        # it to initialize a c++ pointer".  This is useful when you'll be using
        # the "wrap" method to capture a pointer something else generated.
        if not raw:
            self.thisptr = new clib.EarthShape(earthShape)

    def __dealloc__(self):
        if self.thisptr != NULL and not self.owner:
            del self.thisptr

    def getLonDegrees(self, double[:] v):
        """
        Convert a 3-component unit vector to a longitude, in degrees.

        """
        # v is a 1D typed memoryview on an iterable.
        # thispt.getLonDegrees expects a pointer
        # do this by passing the address of the first element, following
        # http://stackoverflow.com/a/14585530/745557
        # XXX: if v is less then 3 elements, this may crash

        return self.thisptr.getLonDegrees(&v[0])

    def getLatDegrees(self, double[:] v):
        """
        Convert a 3-component unit vector to a latitude, in degrees.

        """
        # see above

        return self.thisptr.getLatDegrees(&v[0])

    def getVectorDegrees(self, double lat, double lon):
        """
        Convert geographic lat, lon into a geocentric unit vector. The
        x-component points toward lat,lon = 0, 0. The y-component points toward
        lat,lon = 0, 90. The z-component points toward north pole.

        """
        # thisptr.getVectorDegrees wants two doubles and a pointer to an array
        # that will be filled in c++. we must create a Python object here 
        # that can be returned, and whos memory can be managed by Python, that
        # can be filled in c++ by passing its pointer, following
        # http://docs.cython.org/src/tutorial/array.html#zero-overhead-unsafe-access-to-raw-c-pointer

        # cdef array.array v = array.array('d', [0.0, 0.0, 0.0])
        # self.thisptr.getVectorDegrees(lat, lon, &v.data.as_doubles[0])

        cdef np.ndarray[double, ndim=1, mode="c"] v = np.empty(3)

        # XXX: the commented syntax is preferred, but not working.
        # error: Cannot convert Python object to 'double *'
        # self.thisptr.getVectorDegrees(lat, lon, &v[0] )

        # this syntax works, but isn't preferred.
        # https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC#other-options
        self.thisptr.getVectorDegrees(lat, lon, <double*> v.data)

        return v

    @staticmethod
    cdef EarthShape wrap(clib.EarthShape *cptr, owner=None):
        """
        Wrap a C++ pointer with a pointer-less Python EarthShape class.

        """
        cdef EarthShape inst = EarthShape(raw=True)
        inst.thisptr = cptr
        if owner:
            inst.owner = owner

        return inst


cdef class GeoTessModel:
    """
    GeoTessModel accepts a grid file name and GeoTessMetaData instance.  The
    metadata is _copied_ into the GeoTessModel, so be warned that changes to
    it are _not_ reflected in the original instances.  This is done to
    simplify the life cycle of the underlying C++ memory, because GeoTessModel
    wants to assumes ownership of the provided C++ objects, including
    destruction.

    """
    # XXX: pointer ownership is an issue here.
    # https://groups.google.com/forum/#!searchin/cython-users/$20$20ownership/cython-users/2zSAfkTgduI/wEtAKS_KHa0J
    cdef clib.GeoTessModel *thisptr

    def __cinit__(self, gridFileName=None, GeoTessMetaData metaData=None):
        cdef clib.GeoTessMetaData *md

        if gridFileName is None and metaData is None:
            self.thisptr = new clib.GeoTessModel()
        else:
            if sum((gridFileName is None, metaData is None)) == 1:
                raise ValueError("Must provide both gridFileName and metaData")

            # https://groups.google.com/forum/#!topic/cython-users/6I2HMUTPT6o
            md = metaData.thisptr.copy()
            self.thisptr = new clib.GeoTessModel(gridFileName, md)

    def __dealloc__(self):
        # XXX: doing "del model" still crashes Python.  Dunno why yet.
        if self.thisptr != NULL:
            del self.thisptr

    # https://groups.google.com/forum/#!topic/cython-users/6I2HMUTPT6o

    def loadModel(self, const string& inputFile, relGridFilePath=""):
        # https://groups.google.com/forum/#!topic/cython-users/4ecKM-p8dPA
        if os.path.exists(inputFile):
            self.thisptr.loadModel(inputFile, relGridFilePath)
        else:
            raise exc.GeoTessFileError("Model file not found.")

    def writeModel(self, const string& outputFile):
        self.thisptr.writeModel(outputFile)

    def toString(self):

        return self.thisptr.toString()

    def getEarthShape(self):
        shp = EarthShape.wrap(&self.thisptr.getEarthShape(), owner=self)

        return shp

    def getMetaData(self):
        md = GeoTessMetaData.wrap(&self.thisptr.getMetaData())
        md.owner = self

        return md

    def getGrid(self):
        grid = GeoTessGrid.wrap(&self.thisptr.getGrid())
        grid.owner = self

        return grid

    def setProfile(self, int vertex, int layer, vector[float] &radii, vector[vector[float]] &values):
        """
        Set profile values at a vertex and layer.

        Parameters
        ----------
        vertex, layer : int
            vertex and layer number of the profile.
        radii : list
            Radius values of profile data.
        values : list of lists
            List of corresponding attribute values at the provided radii.

        """
        # holycrap, vector[vector[...]] can just be a list of lists
        # I wonder if it can be a 2D NumPy array
        # TODO: accept NumPy vectors instead of lists for radii and values
        self.thisptr.setProfile(vertex, layer, radii, values)

    def getProfile(self, int vertex, int layer):
        # TODO: return a numpy structured array, not a profile object
        # Just use the Profile object internally here
        pass

    def getNLayers(self):
        returns self.thisptr.getNLayers()

    def getNVertices(self):
        returns self.thisptr.getNVertices()
        

cdef class AK135Model:
    cdef clib.AK135Model *thisptr

    def __cinit__(self):
        self.thisptr = new clib.AK135Model()

    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr

    def getLayerProfile(self, const double &lat, const double &lon, const int &layer):
        cdef vector[float] r
        cdef vector[vector[float]] nodeData

        self.thisptr.getLayerProfile(lat, lon, layer, r, nodeData)

        cdef np.ndarray[double, ndim=1, mode="c"] np_r = np.array(r)
        cdef np.ndarray[double, ndim=2, mode="c"] np_nodeData = np.array(nodeData)

        return np_r, np_nodeData
