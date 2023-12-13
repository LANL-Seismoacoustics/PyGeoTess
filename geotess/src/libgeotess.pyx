#distutils: language = c++
#cython: embedsignature=True
#cython: language_level=3
#cython: c_string_type=unicode
#cython: c_string_encoding=utf-8
import os

# from cpython cimport array
# import array

import numpy as np
cimport numpy as np

np.import_array()

from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray

from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from libcpp.limits cimport numeric_limits
from libcpp.set cimport set
from libcpp cimport bool

cimport clibgeotess as clib
import geotess.exc as exc


cdef class GeoTessData:
    cdef clib.GeoTessData *thisptr

     def __cinit__(self):
        self.thisptr = new clib.GeoTessData()

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr

    # Getters
    def get_double(self, int attributeIndex):
        return self.thisptr.getDouble(attributeIndex)

    def get_float(self, int attributeIndex):
        return self.thisptr.getFloat(attributeIndex)

    def get_long(self, int attributeIndex):
        return self.thisptr.getLong(attributeIndex)

    def get_int(self, int attributeIndex):
        return self.thisptr.getInt(attributeIndex)

    def get_short(self, int attributeIndex):
        return self.thisptr.getShort(attributeIndex)

    def get_byte(self, int attributeIndex):
        return self.thisptr.getByte(attributeIndex)

    # Setters
    def set_value_double(self, int attributeIndex, double value):
        self.thisptr.setValue(attributeIndex, value)

    def set_value_float(self, int attributeIndex, float value):
        self.thisptr.setValue(attributeIndex, value)

    def set_value_long(self, int attributeIndex, int64_t value):
        self.thisptr.setValue(attributeIndex, value)

    def set_value_int(self, int attributeIndex, int value):
        self.thisptr.setValue(attributeIndex, value)

    def set_value_short(self, int attributeIndex, short value):
        self.thisptr.setValue(attributeIndex, value)

    def copy(self):
        cdef clib.GeoTessData new_data = new clib.GeoTessData()
        new_data.thisptr = self.thisptr.copy()
        return new_data


    @staticmethod
    def create_data_from_double_array(values):
        cdef double* c_values = &values[0]
        cdef int size = len(values)
        return clib.GeoTessData.getData(c_values, size)

    @staticmethod
    def create_data_from_float_array(values):
        cdef float* c_values = &values[0]
        cdef int size = len(values)
        return clib.GeoTessData.getData(c_values, size)

cdef class GeoTessUtils:
    """
    Collection of static functions to manipulate geographic information.

    The Utils class provides basic static utility functions for GeoTess to manipulate geographic information.

    """
    # These are almost all static return values in C++, which makes this class
    # more like a module of functions instead of a class with methods.
    cdef clib.GeoTessUtils *thisptr

    def __cinit__(self):
        self.thisptr = new clib.GeoTessUtils()

    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr

    @staticmethod
    def getLatDegrees(double[:] v):
        """
        Convert a 3-component unit vector to geographic latitude, in degrees.
        Uses the WGS84 ellipsoid.

        Parameters
        ----------
        v : array_like
            3-component unit vector of floats

        Returns
        -------
        float
            Geographic latitude in degrees.

        Notes
        -----
        Input arrays longer than length three ignore the remaining values.

        """
        # double getLatDegrees(const double *const v)
        # I think I implemented it with cython "typed memory views.
        # The "double[:]" signifies a 1D Python buffer of doubles
        # "v" shares memory with the Python array-like object.
        # I think the &v[0] can be understood as ‘the address of the first element
        # of the memoryview’. For contiguous arrays, this is equivalent to the start
        # address of the flat memory buffer
        # https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
        return clib.GeoTessUtils.getLatDegrees(&v[0])

    @staticmethod
    def getLonDegrees(double[:] v):
        """
        Convert a 3-component unit vector to geographic longitude, in degrees.
        Uses the WGS84 ellipsoid.

        Parameters
        ----------
        v : array_like
            3-component unit vector

        Returns
        -------
        float
            Geographic longitude in degrees.

        Notes
        -----
        Input arrays longer than length three ignore the remaining values.

        """
        # Cython didn't like "double[:3] v" to clarify it's size
        return clib.GeoTessUtils.getLonDegrees(&v[0])

    @staticmethod
    def getVectorDegrees(double lat, double lon):
        #def getVectorDegrees(const double &lat, const double &lon):
        """ Convert geographic lat, lon into a geocentric unit vector.

        The x-component points toward lat,lon = 0, 0. The y-component points
        toward lat,lon = 0, 90. The z-component points toward north pole. Uses
        the WGS84 ellipsoid.

        Parameters
        ----------
        lat, lon : float

        Returns
        -------
        numpy.ndarray
            x, y, z floats

        """
        # We supply lat, lon Python floats.  Because we supply C types for numeric
        # Python inputs, Cython automatically converts them to their equivalent C types.
        # Addresses to the C-level lat, lon doubles are automatically obtained when passed to
        # GeoTessUtils.getVectorDegrees (I think).

        # C++ returns pointer to 3-vector, but can also take a pointer to the 3-vector.
        # I allocate the output in NumPy, cast it to a typed memoryview, and send its
        # pointer to the method call.
        # TODO: I don't yet know about memory ownership here.
        arr = np.empty(3, dtype=np.float64)
        cdef double[::1] arr_memview = arr
        cdef double* v = clib.GeoTessUtils.getVectorDegrees(lat, lon, &arr_memview[0])

        return arr


    @staticmethod
    def getEarthRadius(double[:] v):
        """
        Retrieve the radius of the Earth in km at the position specified by an
        Earth-centered unit vector. Uses the WGS84 ellipsoid.

        Parameters
        ----------
        v : array_like
            3-element unit vector

        Returns
        -------
        float
            Radius of the Earth in km at specified position.

        """
        return clib.GeoTessUtils.getEarthRadius(&v[0])


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

    def getDescription(self):
        self.thisptr.getDescription()

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

    def getModelSoftwareVersion(self):
        return self.thisptr.getModelSoftwareVersion()

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

        return arr.tolist() # copies the data to a list.

    def getNLayers(self):
        return self.thisptr.getNLayers()

    def getLayerName(self, const int &layerIndex):
        return self.thisptr.getLayerName(layerIndex)

    def getLayerIndex(self, layerName):
        return self.thisptr.getLayerIndex(layerName)

    def getModelFileFormat(self):
        return self.thisptr.getModelFileFormat()

    def setModelFileFormat(self, version):
        self.thisptr.setModelFileFormat(version)


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

    def __cinit__(self, gridFileName=None, GeoTessMetaData metaData=None, viewCopyRight=False):
        cdef clib.GeoTessMetaData *md

        if gridFileName is None and metaData is None:
            self.thisptr = new clib.GeoTessModel()
        else:
            if sum((gridFileName is None, metaData is None)) == 1:
                raise ValueError("Must provide both gridFileName and metaData")

            # https://groups.google.com/forum/#!topic/cython-users/6I2HMUTPT6o
            md = metaData.thisptr.copy()
            self.thisptr = new clib.GeoTessModel(gridFileName, md)

        horizontalType = "LINEAR"
        radialType = "LINEAR"

        if viewCopyRight:
            self.__viewCopyRight()

    @staticmethod
    def __viewCopyRight():
        print("PyGeoTess Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.")
        print("\n")
        GeoTessModel.__viewLANLCopyRight()
        print("\n")
        print("Set viewCopyRight=False to supress this message.")
        return

    @staticmethod
    def __viewLANLCopyRight():
        copyRightString = """
        Copyright (c) 2016, Los Alamos National Security, LLC
        All rights reserved.

        Copyright 2016. Los Alamos National Security, LLC. This software was produced
        under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National
        Laboratory (LANL), which is operated by Los Alamos National Security, LLC for
        the U.S. Department of Energy. The U.S. Government has rights to use,
        reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS
        NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
        LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
        derivative works, such modified software should be clearly marked, so as not to
        confuse it with the version available from LANL.

        BSD Open Source License.

        Additionally, redistribution and use in source and binary forms, with or
        without modification, are permitted provided that the following conditions are
        met:

        1. Redistributions of source code must retain the above copyright notice, this
           list of conditions and the following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice,
           this list of conditions and the following disclaimer in the documentation
           and/or other materials provided with the distribution.
        3. Neither the name of Los Alamos National Security, LLC, Los Alamos National
           Laboratory, LANL, the U.S. Government, nor the names of its contributors may
           be used to endorse or promote products derived from this software without
           specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS
        'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
        THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
        CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
        OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
        IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
        OF SUCH DAMAGE.)"""
        print(copyRightString)
        return

    def __dealloc__(self):
        # XXX: doing "del model" still crashes Python.  Dunno why yet.
        if self.thisptr != NULL:
            del self.thisptr

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()

    # https://groups.google.com/forum/#!topic/cython-users/6I2HMUTPT6o

    def loadModel(self, const string& inputFile, relGridFilePath=""):
        """
        Loads a geotess model given input file name and relative grid file path (relGridFilePath=)
        """
        # https://groups.google.com/forum/#!topic/cython-users/4ecKM-p8dPA
        if os.path.exists(inputFile):
            self.thisptr.loadModel(inputFile, relGridFilePath)
        else:
            raise exc.GeoTessFileError("Model file not found.")

    def writeModel(self, const string& outputFile):
        """
        Write the model object to a file given file name outputFile
        """
        self.thisptr.writeModel(outputFile)

    def getConnectedVertices(self, int layerid):
        """
        Function fo find which vertices are connected
        if a vertex is not connected, then it won't have a set profile
        Argument:
            layerID: integer layer index
        Returns:
            ndarray of connected vertices at this layer
        """
        if layerid < 0 or layerid >= self.getNLayers():
            print("Error, layerid must be between 0 and {}".format(self.getNLayers()-1))
            return -1
        cdef cv = self.thisptr.getConnectedVertices(layerid)
        nvertices = 0
        for i in cv:
            nvertices += 1
        vertices = np.zeros((nvertices,), dtype='int')
        for idx, i in enumerate(cv):
            vertices[idx] = i

        return vertices

    def getPointLatitude(self, pointIndex):
        """
        Use the pointMap object to find the latitude given a pointIndex value
        """
        ptMap = self.thisptr.getPointMap()
        loc = ptMap.getPointLatLonString(pointIndex)
        floatLocation = [float(x) for x in loc.split()]
        return floatLocation[0]

    def getPointLongitude(self, pointIndex):
        """
        Use the pointMap object to find the longitude given a pointIndex value
        """
        ptMap = self.thisptr.getPointMap()
        loc = ptMap.getPointLatLonString(pointIndex)
        floatLocation = [float(x) for x in loc.split()]
        return floatLocation[1]

    def getPointLocation(self, pointIndex):
        """
        Returns the latitude, longitude, radius, and depth of a point in a model defined by the point index
        """
        ptMap = self.thisptr.getPointMap()
        loc = ptMap.getPointLatLonString(pointIndex)
        floatLocation = [float(x) for x in loc.split()]
        lat = floatLocation[0]
        lon = floatLocation[1]
        depth = self.getPointDepth(pointIndex)
        radius = self.getPointRadius(pointIndex)
        return lat, lon, radius, depth

    def getPointVertex(self, pointIndex):
        """
        Returns the vertex given a point index
        """
        ptMap = self.thisptr.getPointMap()
        idx = ptMap.getVertexIndex(pointIndex)
        return idx

    def getPointTessId(self, pointIndex):
        """
        Returns the Tesselation ID given a pointIndex
        """
        ptMap = self.thisptr.getPointMap()
        idx = ptMap.getTessId(pointIndex)
        return idx

    def getPointLayerIndex(self, pointIndex):
        """
        Returns the layer index given a pointIndex
        """
        ptMap = self.thisptr.getPointMap()
        idx = ptMap.getLayerIndex(pointIndex)
        return idx

    def getPointNodeIndex(self, pointIndex):
        """
        Returns the node index (in a profile) given a point index
        """
        ptMap = self.thisptr.getPointMap()
        idx = ptMap.getNodeIndex(pointIndex)
        return idx

    def getPointVertexTessLayerNode(self, pointIndex):
        """
        Parameters
        ----------
        pointIndex : Integer from 0 to self.getNPoints()-1

        Returns
        -------
        ints for: vertex, tessID, layerID, and Node

        """
        ptMap = self.thisptr.getPointMap()
        vertex = ptMap.getVertexIndex(pointIndex)
        tessID = ptMap.getTessId(pointIndex)
        layerID = ptMap.getLayerIndex(pointIndex)
        node = ptMap.getNodeIndex(pointIndex)
        return vertex, tessID, layerID, node

    def getPointData(self, pointIndex):
        """
        For a given point index, returns a vector of attribute values
        """
        ptMap = self.thisptr.getPointMap()
        geotessdata = ptMap.getPointData(pointIndex)
        npts = geotessdata.size()
        dataOut = np.zeros((npts,))
        for i in range(npts):
            dataOut[i] = geotessdata.getDouble(i)
        return dataOut

    def setPointData(self, pointIndex, values):
        """
        For a given pointIndex, sets the values in the GeoTess Model
        """
        ptMap = self.thisptr.getPointMap()
        # below returns a point to values in a point map.
        geoData = ptMap.getPointData(pointIndex)
        for ival, val in enumerate(values):
            # The reference of the pointer is followed in the setter!
            geoData.setValue(ival, val)
        return

    def setPointDataSingleAttribute(self, pointIndex, attributeIndex, value):
        """
        For a given point index and attribute index, sets the value
        """
        ptMap = self.thisptr.getPointMap()
        geoData = ptMap.getPointData(pointIndex)
        geoData.setValue(attributeIndex, value)
        return

    def getNearestPointIndex(self, float latitude, float longitude, float radius):
        """
        Warning! This does not always work. Layer definitions need to be included before it will work properly!
        This is also quite slow.

        Parameters
        ----------
        float latitude :
            floating point from -90 to 90
            Defines the latitude of the lookup point
        float longitude : floating point from -180 to 360
            Defines the longitude of the lookup point.
        float radius : floating point from 0 to ~6371 (earth's radius out from center')
            Defines the radius of the lookup point.

        Returns
        -------
        (int) pointIndex used to map the given location to the nearest point in the tesselation.

        """
        #ptMap = self.thisptr.getPointMap()
        # V2: use the unit vector from the EarthShape class
        ellipsoid = self.getEarthShape()
        inputUnitVector = ellipsoid.getVectorDegrees(latitude, longitude)
        npoints = self.getNPoints()
        # First, loop to get nearest vertex (ie horizontal coordinate, h)
        ptOut = -1
        mindh = 9001
        for pt in range(npoints):
            lat, lon, _, _ = self.getPointLocation(pt)
            testUnitVector = ellipsoid.getVectorDegrees(lat, lon)
            dh = np.linalg.norm(inputUnitVector - testUnitVector)
            if dh < mindh:
                mindh = dh
                vtx = self.getPointVertex(pt)

        # Second, loop to get nearest node (ie vertical coordinate, r)
        # So this is failing when radius is deeper than what is available in connected vertices
        mindr = 9001
        for pt in range(npoints):
            vtmp = self.getPointVertex(pt)
            if vtmp == vtx:
                _, _, rad, _ = self.getPointLocation(pt)
                dr = np.abs(rad - radius)
                if dr < mindr:
                    mindr = dr
                    ptOut = pt

        return ptOut

    def getPointDepth(self, pointIndex):
        """
        Given a point index, return the depth
        """
        cdef float depth
        depth = self.thisptr.getDepth(pointIndex)
        return depth

    def getPointRadius(self, pointIndex):
        """
        Given a point index, return the radius
        """
        cdef float radius
        radius = self.thisptr.getRadius(pointIndex)
        return radius

    def getPointIndex(self, vertex, layer, node):
        """
        Given a vertex, layer, and node, returns the point index
        """
        ptMap = self.thisptr.getPointMap()
        pt = ptMap.getPointIndex(vertex, layer, node)
        return pt

    def getPointIndexLast(self, vertex, layer):
        """
        Returns the point index of the shallowest node in the profile defined by vertex and layer
        """
        ptMap = self.thisptr.getPointMap()
        pt = ptMap.getPointIndexLast(vertex, layer)
        return pt

    def getPointIndexFirst(self, vertex, layer):
        """
        Returns the point index of the deepest node in the profile defined by vertex and layer
        """
        ptMap = self.thisptr.getPointMap()
        pt = ptMap.getPointIndexFirst(vertex, layer)
        return pt

    def toString(self):
        """
        Calls the toString() method
        """
        return self.thisptr.toString()

    def getEarthShape(self):
        """
        Returns the earthshape object
        """
        shp = EarthShape.wrap(&self.thisptr.getEarthShape(), owner=self)
        return shp

    def getMetaData(self):
        """
        returns the metadata object
        """
        md = GeoTessMetaData.wrap(&self.thisptr.getMetaData())
        md.owner = self

        return md

    def getNAttributes(self):
        """
        Returns the number of attributes in the metadata
        """
        md = self.getMetaData()
        att = md.getAttributeNamesString()
        x = att.split()
        return len(x)

    def getGrid(self):
        """
        Returns the grid object
        """
        #XXX: I don't think this works
        grid = GeoTessGrid.wrap(&self.thisptr.getGrid())
        grid.owner = self

        return grid

#     def setProfile(self, int vertex, int layer, vector[float] &radii, vector[vector[float]] &values):
#         """
#         Set profile values at a vertex and layer.
#         This version works with c++ style vector types.
#         Use setProfileND to push ndarrays instead.

#         Parameters
#         ----------
#         vertex, layer : int
#             vertex and layer number of the profile.
#         radii : list
#             Radius values of profile data.
#         values : list of lists
#             List of corresponding attribute values at the provided radii.

#         Returns:
#             1 on success
#             -1 on failure

#         """

#         try:
#             self.thisptr.setProfile(vertex, layer, radii, values)
#             return 1
#         except:
#             return -1

    def setProfile(self, int vertex, int layer, 
                   profile=None, radii=None, values=None, 
                   nRadii=0, nNodes=0, nAttributes=0):
        """
        Set profile values at a vertex and layer in the GeoTessModel.
        This method is overloaded to handle different types of inputs.

        Parameters:
        vertex (int): Vertex index in the model.
        layer (int): Layer index in the model.
        profile (GeoTessProfile or None): GeoTessProfile object. Default is None.
        radii (list or None): List of radius values. Default is None.
        values (list or None): List of attribute values. Default is None.
        nRadii, nNodes, nAttributes (int): Sizes for radii, values, and attributes arrays.

        Raises:
        ValueError: If the arguments provided are invalid.
        """
        cdef vector[float] c_radii
        cdef vector[vector[float]] c_values

        if profile is not None:
            c_profile = profile.thisptr if profile else NULL
            self.thisptr.setProfile(vertex, layer, c_profile)
        elif radii is not None and values is not None:
            c_radii = radii
            c_values = values
            self.thisptr.setProfile(vertex, layer, c_radii, c_values)
        elif radii is not None:
            self.thisptr.setProfile(vertex, layer, <float*>radii, nRadii, <float**>values, nNodes, nAttributes)
    else:
        raise ValueError("Invalid arguments for setProfile")

    
            
            
            
    def setSurfaceProfile(self, int vertex, int layer, GeoTessData data):
        """
        Sets a surface profile for a given vertex and layer in the GeoTessModel.

        Parameters:
        vertex (int): The vertex index in the 2D grid.
        layer (int): The layer index in the model.
        data (GeoTessData): Object containing the data to be set.
        """
        cdef clib.GeoTessProfileSurface* profile_surface
        profile_surface = new clib.GeoTessProfileSurface(data.thisptr)
        self.thisptr.setProfile(vertex, layer, <clib.GeoTessProfile*>profile_surface)



    def setProfileND(self, int vertex, int layer, radii, values):
        """
        Set profile values at a vertex and layer using ndarrays rather than c++ vector types

        Parameters
        ----------
        int vertex, layer
            vertex and layer indices of the profile
        radii : 1D ndarray
            ndarray radius values of the profile data
        values : 2D ndarray
            nradii x nattributes ndarray of attribute values at the provided radii

        Returns:
            1 on success
            -1 on values not being 2D ndarray
            -2 on errors packing ndarray in c++ vectors
            -3 on error setting profile values
        """
        cdef vector[float] cradii
        cdef vector[vector[float]] cvalues
        cdef vector[float] ctmp

        # Radii have to increase.
        # Put a check here to make sure the input radii and values ndarrays
        # are in increasing radius, that is radius outward from the center
        # of the earth
        if radii[1] < radii[0]:
            tmp = np.flipud(radii)
            radii = tmp.copy()
            tmp = np.flipud(values)
            values = tmp.copy()

        try:
            (nr, na) = values.shape
        except:
            print("Error in setProfileND: values must be nradii x nattributes ndarray")
            return -1
        try:
            cradii.reserve(nr)
            for ir, r in enumerate(radii):
                cradii.push_back(r)
                ctmp.clear()
                for ia, a in enumerate(values[ir]):
                    ctmp.push_back(a)
                cvalues.push_back(ctmp)
        except:
            print("Error in setProfileND: c++ vector fill error")
            return -2
        try:
            self.thisptr.setProfile(vertex, layer, cradii, cvalues)
            cradii.clear()
            cvalues.clear()
            return 1
        except:
            print("Error in setProfileND: c++ call failed.")
            return -3


    def getProfileTypeInt(self, int vertex, int layer):
        """
        Given a vertex and layer, returns the profile type as an integer
        """
        A = self.thisptr.getProfile(vertex, layer)
        return A.getTypeInt()


    def getProfile(self, int vertex, int layer):
        """
        Gets values in a profile given the vertex and layer.
        returns nradius x 1 radius vector and nradius x nattributes attributes matrix
        """
        nv = self.getNVertices()
        if vertex >= nv or vertex < 0:
            print("Error, vertex {} outside of range (0 - {})".format(vertex, nv-1))
            return -1, -1
        nl = self.getNLayers()
        if layer >= nl or layer < 0:
            print("Error, layer {} outside of range (0 - {})".format(layer, nl-1))
            return -2, -2
        cdef float *r
        A = self.thisptr.getProfile(vertex, layer)
        nradii = A.getNRadii()
        ndata = A.getNData()
        r = A.getRadii()
        nparams = self.getNAttributes()
        radiusPy = np.zeros((nradii,))
        if ndata > 0:
            attributesPy = np.zeros((nradii, nparams))
            for idx in range(nradii):
                B = A.getData(idx)
                for jdx in range(B.size()):
                    attributesPy[idx, jdx] = B.getDouble(jdx)
                radiusPy[idx] = r[idx]
        else:
            for idx in range(nradii):
                radiusPy[idx] = r[idx]
            attributesPy = None
        return radiusPy, attributesPy

    def getNLayers(self):
        """
        Returns the number of layers.
        """
        return self.thisptr.getNLayers()

    def getNVertices(self):
        """
        Returns the number of vertices.
        """
        return self.thisptr.getNVertices()

    def getNPoints(self):
        """
        Returns the number of points
        """
        return self.thisptr.getNPoints()

    def getNRadii(self, int vertex, int layer):
        """
        For a given vertex and layer, returns the number of radii (nodes)
        """
        return self.thisptr.getNRadii(vertex, layer)

    def getWeights(self, const double[::1] pointA, const double[::1] pointB, const double pointSpacing, const double radius, str horizontalType):
        """ Compute the weights on each model point that results from interpolating positions along the specified ray path.

        This method is only applicable to 2D GeoTessModels.

        Parameters
        ----------
        pointA, pointB : array_like
            The 3-element unit vector of floats defining the beginning, end of the great circle path
            C-contiguous layout of floats.
        pointSpacing : float
            The maximum spacing between points, in radians. The actual spacing will generally be
            slightly less than the specified value in order for there to be an integral number of
            uniform intervals along the great circle path.
        radius : float
            The radius of the great circle path, in km. If the value is less than or equal to zero
            then the radius of the Earth determined by the current EarthShape is used.
            See getEarthShape() and setEarathShape() for more information about EarthShapes.
        horizontalType : str {'LINEAR', 'NATURAL_NEIGHBOR'}

        Returns
        -------
        weights : dict
            Integer keys to float values. (output) map from pointIndex to weight.
            The sum of the weights will equal the length of the ray path in km.

        Notes
        -----
        The following procedure is implemented:
        1. divide the great circle path from pointA to pointB into nIntervals which each are of length less than or equal to pointSpacing.
        2.  multiply the length of each interval by the radius of the earth at the center of the interval, which converts the length of the interval into km.
        3. interpolate the value of the specified attribute at the center of the interval.
        4. sum the length of the interval times the attribute value, along the path.

        """
        # TODO: make this return two NumPy arrays instead?

        # pointA and pointB were specified as typed memorviews, so we can send the address to their first
        # element as the double pointer.
        # pointSpacing and radius are automatically converted to addresses by Cython because they
        # are numeric types and we specified them as typed in the calling signature.
        # The output of the C++ getWeights method is a map, which will automatically be converted to
        # a Python dict by Cython.

        cdef const clib.GeoTessInterpolatorType* interpolator
        cdef cmap[int, double] weights

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            interpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        self.thisptr.getWeights(&pointA[0], &pointB[0], pointSpacing, radius,
                                deref(interpolator),
                                weights)

        return weights

    def getValueFloat(self, int pointIndex, int attributeIndex):
        """
        For a given point index and attribute index, returns the value
        """
        return self.thisptr.getValueFloat(pointIndex, attributeIndex)

    # Series of position methods. They start with defining the interpolator types
    def positionToString(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns a string for a position object given latitude, longitude, and depth
        optionally, give horizontalType and/or radialType interpolators
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        return str(pos.toString())

    def positionToStringLayer(self, layerid, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns a string for a position object given layerid, latitude, longitude, and depth
        optionally, give horizontalType and/or radialType interpolators
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(layerid, lat, lon, depth)
        return str(pos.toString())

    def positionGetLayer(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        returns the layerID as a function of latitude, longitude, and depth.
        Optionally, give position interpolation methods horizontalType and/or radialType
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        R = pos.getEarthRadius()
        radius = R-depth
        layid = pos.getLayerId(radius)
        return layid

    def positionGetVector(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        For a given latitude, longitude, and depth, get the position vector
        Optionally, give horizontalType and/or radialType interpolators
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        cdef double* vec = pos.getVector()
        output = np.zeros((3,))
        output[0] = vec[0]
        output[1] = vec[1]
        output[2] = vec[2]
        return output

    def positionGetRadiusBottomLayer(self, layer, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Finds the bottom radius (nearest the core) for a position object
        defined by location and layer

        Parameters
        ----------
        layer : int
            layer index.
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
        radius (km) at bottom of layer.

        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(layer, lat, lon, depth)
        rad = pos.getRadiusBottom(layer)
        return rad

    def positionGetRadiusTopLayer(self, layer, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Finds the top radius (nearest the surface) for a position object
        defined by location and layer

        Parameters
        ----------
        layer : int
            layer index.
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.

        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
        radius (km) at top of layer.

        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(layer, lat, lon, depth)
        rad = pos.getRadiusTop(layer)
        return rad

    def positionGetValue(self, lat, lon, depth, attribute, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns the attribute at a position

        Parameters
        ----------
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        attribute: int
            attribute index
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
        attribute value at position.
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        val = pos.getValue(attribute)
        return val

    def positionGetValueLayer(self, layer, lat, lon, depth, attribute, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns the attribute at a position, but forces it to be in layer

        Parameters
        ----------
        layer: int
            layer index
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        attribute: int
            attribute index
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
        attribute value at position.
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(layer, lat, lon, depth)
        val = pos.getValue(attribute)
        return val

    def positionGetValues(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns the attribute values at a position

        Parameters
        ----------
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
            ndarray of attribute values at position
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        nattributes = self.getNAttributes()
        values = np.zeros((nattributes,))
        for iatt in range(nattributes):
            values[iatt] = pos.getValue(iatt)
        return values

    def positionGetValuesLayer(self, layer, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns the attribute at a position, but forces it to be in layer

        Parameters
        ----------
        layer: int
            layer index
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
            ndarray of attribute values at position
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(layer, lat, lon, depth)
        nattributes = self.getNAttributes()
        values = np.zeros((nattributes,))
        for iatt in range(nattributes):
            values[iatt] = pos.getValue(iatt)
        return values

    def positionGetTriangle(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns which triangle number the given location is located within.

        Parameters
        ----------
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
            Integer triangle where position is located
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        tri = pos.getTriangle()
        return tri

    def positionGetIndexOfClosestVertex(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns the closest vertex to the given location

        Parameters
        ----------
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
            integer vertex
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        idx = pos.getIndexOfClosestVertex()
        return idx

    def positionGetIndexOfClosestVertexLayer(self, layerid, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns the closest vertex to the given location and layer

        Parameters
        ----------
        layerid : integer
            layer index
        lat : float
            latitude.
        lon : float
            longitude.
        depth : float
            depth from surface of ellipsoid.
        Optionally, give horizontalType and/or radialType interpolators

        Returns
        -------
            integer vertex
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(layerid, lat, lon, depth)
        idx = pos.getIndexOfClosestVertex()
        return idx

    def positionGetDepth(self, lat, lon, radius, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Most position methods take depth. This method takes radius and converts to depth for the model's ellipsoid

        Parameters
        ----------
        lat : float
            latitude.
        lon : float
            longitude.
        radius : float
            radius from center of earth (km).

        Returns
        -------
        depth from surface of the earth (km).

        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        dtmp = 6380-radius
        pos.set(lat, lon, dtmp)
        R = pos.getEarthRadius()
        depth = R - radius
        return depth

    def positionGetRadius(self, lat, lon, depth, horizontalType="LINEAR", radialType="LINEAR"):
        """
        determines radius from input depth

        Parameters
        ----------
        lat : float
            latitude (deg).
        lon : float
            longitude (deg).
        depth : float
            depth from ellipsoid surface (km).

        Returns
        -------
        radius from center of earth (km).

        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))
        pos.set(lat, lon, depth)
        R = pos.getEarthRadius()
        radius = R-depth
        return radius

    def positionGetBorehole(self, float lat, float lon, float dz=10.0, computeDepth = False, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Returns layerID vector, radii vector, and attribute matrix for the given latitude, longitude position

        Parameters
        ----------
        lat : float
            latitude.
        lon : float
            longitude.
        dz : float
            regular depth sampling, km, for the borehole
        Optionally, give horizontalType and/or radialType interpolators
        set computeDepth=True to convert output from radii to depth

        Returns
        -------
            vector layers, vector radii, matrix attributes
        """
        cdef vector[int] layers
        cdef vector[double] radii
        cdef vector[double] attributes
        R = self.positionGetRadius(lat, lon, 0)
        npts = int(np.ceil(R/dz))
        layers.reserve(npts)
        radii.reserve(npts)
        nattributes = self.getNAttributes()
        attributes.reserve(npts * nattributes)

        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))

        pos.set(lat, lon, 0)
        if computeDepth:
            computeDepthFlag = 1
        else:
            computeDepthFlag = 0
        i = pos.getBorehole(dz, computeDepthFlag, layers, radii, attributes)
        layersOut = np.zeros((layers.size(),))
        radiiOut = np.zeros((radii.size(),))
        attributesOut = np.zeros((radii.size(), nattributes))
        for idx in range(layers.size()):
            layersOut[idx] = layers[idx]
            radiiOut[idx] = radii[idx]
            for j in range(nattributes):
                jdx = j + idx * nattributes
                attributesOut[idx, j] = attributes[jdx]
        return layersOut, radiiOut, attributesOut



    def getGeographicLocationAttribute(self, float lat, float lon, float radius, int attribute, int layer, float dz=1.0, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Uses interpolation to lookup the value of an attribute at a point given latitude, longitude, radius, attribute index, and layer index
        Optionally give dz for depth search to check the layer
        Optionally give horizontalType and/or radialType interpolators
        On success, returns a single value.
        On failure, returns None object.
        """
        cdef const clib.GeoTessInterpolatorType* horizontalInterpolator
        cdef const clib.GeoTessInterpolatorType* radialInterpolator

        if horizontalType in ('LINEAR', 'NATURAL_NEIGHBOR'):
            horizontalInterpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        else:
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        if radialType in ('LINEAR', 'CUBIC_SPLINE'):
            radialInterpolator = clib.GeoTessInterpolatorType.valueOf(radialType)
        else:
            msg = "radialType must be either 'LINEAR' or 'CUBIC_SPLINE'."
            raise ValueError(msg)
        pos = self.thisptr.getPosition(deref(horizontalInterpolator), deref(radialInterpolator))

        # So this is a little weird because we can't just set a depth, but need to know what layer to look in as well
        depth = self.positionGetDepth(lat, lon, radius, horizontalType=horizontalType, radialType=radialType)
        pos.set(layer, lat, lon, depth)

        if layer > self.getNLayers():
            return None
        else:
            try:
                rbot = pos.getRadiusBottom(layer)
                rtop = pos.getRadiusTop(layer)
                nradii = np.round((rtop - rbot)/dz)
                if nradii < 2:
                    nradii = 2
                dr = (rtop - rbot) / (nradii-1)
                offset = 9999.0
                tmprad = 0.0
                for i in range(int(nradii)):
                    r = rbot + i * dr
                    if np.abs(radius - r) < offset:
                        offset = np.abs(radius-r)
                        tmprad = r
                pos.setRadius(layer, tmprad)
                v = pos.getValue(attribute)
                return v
            except:
                return None




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


cdef class GeoTessModelAmplitude(GeoTessModel):
    """
    Amplitude extension class of GeoTessModel.

    """
    cdef clib.GeoTessModelAmplitude *thisampptr

    def __cinit__(self, modelInputFile=None):
        # GeoTessModelAmplitude() is now protected so we can't use it here.
        if modelInputFile is None:
            self.thisampptr = new clib.GeoTessModelAmplitude() #removed new
        else:
            self.thisampptr = new clib.GeoTessModelAmplitude(modelInputFile) #removed new
    
    def __dealloc__(self):
        if self.thisampptr != NULL:
            del self.thisampptr

    def getSiteTrans(self, const string& station, const string& channel, const string& band):
        """ Retrieve the site term for the specified station/channel/band or NaN if not supported.

        Parameters
        ----------
        station, channel, band : str

        Returns
        -------
        float or None
            Site term.

        """
        cdef float site_trans = self.thisampptr.getSiteTrans(station, channel, band)

        # from CPPGlobals.h, I see GeoTess uses quiet_NaN as "NaN_FLOAT",
        # cast to a float.  Not sure this comparison will work.
        # Saw this in https://github.com/cython/cython/blob/master/tests/run/libcpp_all.pyx
        cdef float NaN_FLOAT = numeric_limits[float].quiet_NaN()
        if site_trans == NaN_FLOAT:
            out = None
        else:
            out = site_trans

        return out

    def getPathCorrection(self, const string& station, const string& channel, const string& band,
            const double& rcvLat, const double& rcvLon,
            const double& sourceLat, const double& sourceLon):
        '''Retrieve Q effect on amplitude for a specified source-receiver path. 
        or NaN if not supported.
            
                    Parameters
                    ----------
                    station : str, channel : str, band : str, rcvLat : float, rcvLon : float, 
                                sourceLat : float, sourceLon  : float
                    Returns
                    -------
                    double or None
                        Path Correction.'''

        cdef double path_correction = self.thisampptr.getPathCorrection(station, channel, band, rcvLat, rcvLon, sourceLat, sourceLon)


        return path_correction

cdef class GeoTessData:
    cdef clib.GeoTessData *thisptr

    def __cinit__(self):

        self.thisptr = new clib.GeoTessData()

    def __dealloc__(self):
        # Freeing the C++ object when the Python object is deleted
        if self.thisptr is not NULL:
            del self.thisptr

    def get_double(self, int attributeIndex):
        return self.thisptr.getDouble(attributeIndex)
