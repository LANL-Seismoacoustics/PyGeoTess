#distutils: language = c++
#cython: embedsignature=True
"""
This module exposes Cython GeoTess functionality from the pxd file into Python.

The class definitions here are Python-visible, and are simply wrappers that 
forward the Python-exposed methods directly to their Cython-exposed C++
counterparts, which have been exposed in the imported pxd file.

This module is also responsible for converting between Python types and c++
types, which sometimes involves annoying tricks.  For simple numerical types,
this conversion can be done automatically in the calling signature of a "def"
method if types are declared.  Complex C++ class types, for example, can't be
in a Python-visable "def" method because Python objects can't be automatically
cast to C++ types.  For these cases, sneaky factory functions that can used
accept the complex types must do the work.  Unfortunately, this means that any
constructor or method that accepts complex c++ can't be "directly" exposed to
Python.

Using both a pxd and a pyx file is done, partly, so that we can keep the
exposed c++ GeoTess functionality together in one namespace using "cimport",
so that we can name the classes exposed to Python the same as those in the
GeoTess c++.  This is sometimes confusing in error messages, however.

GeoTess functionality is intentionally a one-to-one translation into Python so
that any modifications to the way models and grids are used can be developed
and tested in in pure Python modules.  This makes it easier to try different
Python approaches to working with the underlying GeoTess library.


## Current conversion conventions

* GeoTess unit vectors are returned as 3-element of NumPy arrays of doubles


## Current headaches

* Deleting or garbage-collecting objects is dangerous.  Some objects are
  managed by other objects, so deleting them manually can crash the interpreter.
  I'm not sure how to fix this yet.

"""
import os

# from cpython cimport array
# import array

import numpy as np
cimport numpy as np

np.import_array()

from cython.operator cimport dereference as deref

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

    def __cinit__(self):
        self.thisptr = new clib.GeoTessGrid()

    def __dealloc__(self):
        if self.thisptr != NULL:
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
 
    def getNTriangles(self):
        return self.thisptr.getNTriangles()

    def getNTessellations(self):
        return self.thisptr.getNTessellations()

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

    def getVertexIndex(self, int triangle, int corner):
        return self.thisptr.getVertexIndex(triangle, corner)


cdef class GeoTessMetaData:
    cdef clib.GeoTessMetaData *thisptr

    def __cinit__(self):
        self.thisptr = new clib.GeoTessMetaData()

    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr

    def setEarthShape(self, const string& earthShapeName):
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

    def __cinit__(self, earthShape="WGS84", raw=False):
        # raw=True means "just give me the Python wrapper class, I don't want
        # it to initialize a c++ pointer".  This is useful when you'll be using
        # the "wrap" method to capture a pointer something else generated.
        if not raw:
            self.thisptr = new clib.EarthShape(earthShape)

    def __dealloc__(self):
        if self.thisptr != NULL:
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

        # XXX: this syntax is preferred, but not working
        # error: Cannot convert Python object to 'double *'
        # self.thisptr.getVectorDegrees(lat, lon, &v[0] )

        # https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC#other-options
        self.thisptr.getVectorDegrees(lat, lon, <double*> v.data)

        return v

    @staticmethod
    cdef EarthShape wrap(clib.EarthShape *cptr):
        """
        Wrap a C++ pointer with a pointer-less Python EarthShape class.

        """
        cdef EarthShape inst = EarthShape(raw=True)
        inst.thisptr = cptr

        return inst


cdef class GeoTessModel:
    """
    GeoTessModel accepts a GeoTessGrid and GeoTessMetaData instance.  These
    instances are _copied_ into the GeoTessModel. Be warned that changes to
    them are _not_ reflected in the original instances.  This is done to
    simplify the life cycle of the underlying C++ memory, because GeoTessModel
    wants to assumes ownership of the provided C++ objects, including
    destruction.

    """
    # XXX: pointer ownership is an issue here.
    # https://groups.google.com/forum/#!searchin/cython-users/$20$20ownership/cython-users/2zSAfkTgduI/wEtAKS_KHa0J
    cdef clib.GeoTessModel *thisptr

    def __cinit__(self, gridFileName=None, GeoTessMetaData metaData=None):
        # a cdef can't be inside a conditional statement, otherwise these
        # would be in the else clause.
        # https://groups.google.com/forum/#!topic/cython-users/iNmemRwUyuU
        cdef clib.GeoTessMetaData *mdptr

        if gridFileName is None and metaData is None:
            self.thisptr = new clib.GeoTessModel()
        else:
            if sum((gridFileName is None, metaData is None)) == 1:
                raise ValueError("Must provide both gridFileName and metaData")

            # copy the metadata, so that GeoTessModel can truly control
            # the destruction of the metadata it uses.
            mdptr = new clib.GeoTessMetaData(deref(metaData.thisptr))

            # https://groups.google.com/forum/#!topic/cython-users/6I2HMUTPT6o
            self.thisptr = new clib.GeoTessModel(gridFileName, mdptr)

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
        return EarthShape.wrap(&self.thisptr.getEarthShape())
