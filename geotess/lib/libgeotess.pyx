"""
This module exposes Cython GeoTess functionality from the pxd file into Python.

The class definitions here are Python-visible, and are simply wrappers that
forward the Python-exposed methods directly down to their Cython-exposed C++
counterparts, which have been exposed in the imported pxd file.

This module is also responsible for converting between Python types and C++
types, which sometimes involves annoying tricks.  For simple numerical types,
this conversion can be done automatically in the calling signature of a "def"
method if types are declared.  Complex C++ class types, for example, can't be
in a Python-visible "def" method because Python objects can't be automatically
cast to C++ types.  For these cases, sneaky factory functions that can accept
the complex types must do the work.  Unfortunately, this means that any
constructor or method that accepts complex C++ can't be "directly" exposed to
Python.

Using both a pxd and a pyx file is done, partly, so that we can keep the
exposed C++ GeoTess functionality together in one namespace using "cimport",
such that we can name the classes exposed to Python the same as those in the
GeoTess C++.  This is sometimes confusing in error messages, however.

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
  they're forwarded to the C++ methods.  This is dangerous.

## Original C++ documentation
https://sandialabs.github.io/GeoTessCPP/GeoTessCPP/doc/html/annotated.html

"""
# good page on calling signatures.  Doesn't yet know about typed memoryviews, though.
# https://medium.com/@yusuken/calling-c-functions-from-cython-references-pointers-and-arrays-e1ccb461b6d8
# good page on C++ and cython
# https://azhpushkin.me/posts/cython-cpp-intro
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


cimport geotess.lib.geotesscpp as clib

import geotess.lib as lib
import geotess.exc as exc

cdef class GeoTessGrid:
    """ Manages the geometry and topology of one or more multi-level triangular tessellations of a unit sphere.

    Has many functions to retrieve information about the grid but knows nothing about Data.

    Manages the geometry and topology of one or more multi-level triangular tessellations of a unit sphere.
    It knows:

    * the positions of all the vertices,
    * the connectivity information that defines how vertices are connected to form triangles,
    * for each triangle it knows the indexes of the 3 neighboring triangles,
    * for each triangle it knows the index of the triangle which is a descendant at the next higher
      tessellation level, if there is one.
    * information about which triangles reside on which tessellation level

    GeoTessGrid is thread-safe in that its internal state is not modified after its data has been loaded
    into memory. The design intention is that single instances of a GeoTessGrid object and GeoTessData
    object can be shared among all the threads in a multi-threaded application and each thread will have
    it's own instance of a GeoTessPosition object that references the common GeoTessGrid + GeoTessData
    combination.

    Parameters
    ----------
    raw : bool
        If True, return an "raw" (empty) instance (doesn't initialize its own pointer).
        This is intended for use with c-level classmethods that instantiate an instance
        from an existing pointer.

    """
    cdef clib.GeoTessGrid *thisptr
    cdef object owner
    cdef bint owns_ptr # True if the instance owns its pointer, otherwise likely came from model.getGrid()

    def __cinit__(self, raw=False):
        # XXX: lots of things evaluate to True or False. A file name, for example.
        if not raw:
            self.thisptr = new clib.GeoTessGrid()

    def __dealloc__(self):
        # if self.thisptr != NULL and not self.owner:
        if self.thisptr is not NULL and self.owns_ptr:
            del self.thisptr

    @staticmethod
    cdef GeoTessGrid wrap(clib.GeoTessGrid *cptr, owner=None):
        """
        Wrap a C++ pointer with a pointer-less Python GeoTessGrid class.

        Deprecated.  Use `from_pointer` instead.
        """
        cdef GeoTessGrid inst = GeoTessGrid(raw=True)
        inst.thisptr = cptr
        if owner:
            inst.owner = owner

        return inst

    @staticmethod
    cdef GeoTessGrid from_pointer(clib.GeoTessGrid *cptr):
        """ Initialize GeoTessGrid from a C++ GeoTessGrid pointer.

        The resulting grid instance doesn't own the pointer and won't free its memory
        when deleted or garbage collected.

        """
        # from "Instantiation from existing C/C++ pointers" in Cython docs
        cdef GeoTessGrid wrapper = GeoTessGrid.__new__(GeoTessGrid, raw=True)
        wrapper.thisptr = cptr
        wrapper.owns_ptr = False

        return wrapper

    def loadGrid(self, str inputFile):
        """ Load GeoTessGrid object from a File.

        Parameters
        ----------
        inputFile : str
            name of file from which to load grid.

        Returns
        -------
        pointer to a Grid object

        """
        # TODO: I think this needs to become a staticmethod: grid = GeoTessGrid.loadGrid(gridfilename)
        # I honestly don't know how this works as written.  CPP returns a pointer to a GeoTessGrid, but
        # I don't capture it here.  Somehow it still gets assigned to .thisptr.
        if os.path.exists(inputFile):
            self.thisptr.loadGrid(bytes(inputFile, encoding='utf-8')) # bytes implicitly coerces to std::string
        else:
            raise exc.GeoTessFileError(f"File not found: {inputFile}")

    def writeGrid(self, str fileName):
        self.thisptr.writeGrid(bytes(fileName, encoding='utf-8'))

    def getGridInputFile(self):
        """ Retrieve the name of the file from which the grid was loaded.

        This will be the name of a GeoTessModel file if the grid was stored in the same file as the model.

        Returns
        -------
        str
            the name of the file from which the grid was loaded.

        """
        return self.thisptr.getGridInputFile()

    def testGrid(self):
        """ Tests the integrity of the grid.

        Visits every triangle T, and (1) checks to ensure that every neighbor of T includes T in its list of neighbors, and (2) checks that every neighbor of T shares exactly two nodes with T.

        Exceptions
        ----------
        GeoTessException if anything is amiss.

        """
        self.thisptr.testGrid()

    def getMemory(self):
        """ Retrieve the amount of memory required by this GeoTessGrid object in bytes.

        Returns
        -------
        int
            the amount of memory required by this GeoTessGrid object in bytes.

        """
        return self.thisptr.getMemory()

    def getNLevels(self, tessellation=None):
        """ Returns the number of tessellation levels defined for this grid.

        Parameters
        ----------
        tessellation : int
            Return only number of levels for the provided tessellation index, otherwise
            all levels are returned.

        Returns
        -------
        int
            The number of tessellation levels defined for this grid.

        """
        if tessellation is None:
            nlevels = self.thisptr.getNLevels()
        else:
            nlevels = self.thisptr.getNLevels(tessellation)

        return nlevels

    def getNTriangles(self, tessellation=None, level=None):
        """ Retrieve the number of triangles that define the specified level of the specified
        multi-level tessellation of the model.

        Parameters
        ----------
        tessellation : int
            Specified tessellation index.
        level : int
            index of a level relative to the first level of the specified tessellation

        Returns
        -------
        int
            number of triangles on specified tessellation and level.

        """
        if tessellation is None and level is None:
            NTriangles = self.thisptr.getNTriangles()
        else:
            Nlevels = self.thisptr.getNLevels()
            NTess = self.getNTessellations()
            # TODO: doesn't GeoTessCPP already do this kind of checking?
            if level > Nlevels or tessellation > NTess:
                msg = "level > {} or tessellation > {}".format(Nlevels, NTess)
                raise ValueError(msg)
            NTriangles = self.thisptr.getNTriangles(int(tessellation), int(level))

        return NTriangles

    def getNTessellations(self):
        """ Returns the number of tessellations in the tessellations array.

        Returns
        -------
        int
            the number of multi-level tesseallations.

        """
        return self.thisptr.getNTessellations()

    def getNVertices(self):
        """ Returns the number of vertices in the vectices array.

        Returns
        -------
        int
            number of vertices

        """
        return self.thisptr.getNVertices()

    def getVertices(self):
        """ Retrieve a reference to all of the vertices.

        Vertices consists of an nVertices x 3 array of doubles. The double[3] array associated
        with each vertex is the 3 component unit vector that defines the position of the vertex.

        Users should not modify the contents of the array.

        Returns
        -------
        numpy.ndarray
            nVertices x 3 array of unit vectors.

        """
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

        # free c_vertices b/c data has already been copied?
        return ArgsArray

    def toString(self):
        """ Summary information about this GeoTessGrid object.

        Returns
        -------
        str
            Summary description string for current grid.

        """
        return self.thisptr.toString()

    def getGridID(self):
        #const string& getGridID() const
        # return self.thisptr.getGridID().decode('utf-8')
        return self.thisptr.getGridID()

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
        np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_ARRAY_OWNDATA)
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
        Supply an integer triangle index, get a 3-element numpy integer array, which
        are indices of the vertices that make this triangle.

        """
        cdef const int *tri_vertex_ids = self.thisptr.getTriangleVertexIndexes(triangleIndex)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 3
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> tri_vertex_ids)
        np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_ARRAY_OWNDATA)

        return arr.copy()

    def getFirstTriangle(self, int tessellation, int level):
        """ Retrieve the index of the first triangle on the specified level of the
        specified tessellation of the model.

        Parameters
        ----------
        tessellation : int
        level : int
            index of a level relative to the first level of the specified tessellation

        Returns
        -------
        int
            a triangle index

        """
        return self.thisptr.getFirstTriangle(tessellation, level)

    def getLastTriangle(self, int tessellation, int level):
        """ Retrieve the index of the last triangle on the specified level of the specified
        tessellation of the model.

        Parameters
        ----------
        tessellation : int
        level : int
            index of a level relative to the first level of the specified tessellation

        Returns
        -------
        int
            a triangle index

        """
        return self.thisptr.getLastTriangle(tessellation, level)

    def getVertexIndex(self, int triangle, int corner, tessId=None, level=None):
        """ Get the index of the vertex that occupies the specified position in the hierarchy.

        Parameters
        ----------
        triangle : int
            the i'th triangle in the grid
        corner : int
            the i'th corner of the specified triangle
        tessId : int
            tessellation index
        level : int
            index of a level relative to the first level of the specified tessellation

        Returns
        -------
        int
            index of a vertex

        """
        # XXX: the order of these method arguments doesn't match CPP, but used here for
        # backwards PyGeoTess compatibility.  Supplying them in the wrong order can crash
        # the interpreter :-(
        if tessId is not None and level is not None:
            out = self.thisptr.getVertexIndex(tessId, level, triangle, corner)
        else:
            out = self.thisptr.getVertexIndex(triangle, corner)

        return out



cdef class GeoTessMetaData:
    """ Top level class that manages the GeoTessMetaData, GeoTessGrid and GeoTessData
    that comprise a 3D Earth model.

    GeoTessModel manages the grid and data that comprise a 3D Earth model.
    The Earth is assumed to be composed of a number of layers each of which spans the
    entire geographic extent of the Earth. It is assumed that layer boundaries do not
    fold back on themselves, i.e., along any radial profile through the model, each layer
    boundary is intersected exactly one time. Layers may have zero thickness over some
    or all of their geographic extent. Earth properties stored in the model are assumed
    to be continuous within a layer, both geographically and radially, but may be discontinuous
    across layer boundaries.

    A GeoTessModel is comprised of 3 major components:

    The model grid (geometry and topology) is managed by a GeoTessGrid object. The grid is made
    up of one or more 2D triangular tessellations of a unit sphere.

    The data are managed by a 2D array of Profile objects. A Profile is essentially a list of
    radii and Data objects distributed along a radial profile that spans a single layer at a
    single vertex of the 2D grid. The 2D Profile array has dimensions nVertices by nLayers.

    Important metadata about the model, such as the names of the major layers, the names of
    the data attributes stored in the model, etc., are managed by a GeoTessMetaData object.
    The term 'vertex' refers to a position in the 2D tessellation. They are 2D positions
    represented by unit vectors on a unit sphere. The term 'node' refers to a 1D position
    on a radial profile associated with a vertex and a layer in the model. Node indexes are
    unique only within a given profile (all profiles have a node with index 0 for example).
    The term 'point' refers to all the nodes in all the profiles of the model. There is only one
    'point' in the model with index 0. PointMap is introduced to manage all these different indexes.

    """
    cdef clib.GeoTessMetaData *thisptr
    cdef object owner
    cdef bint owns_ptr

    def __cinit__(self, raw=False):
        if not raw:
            self.thisptr = new clib.GeoTessMetaData()

    def __dealloc__(self):
        if self.thisptr != NULL and not self.owner:
            del self.thisptr #XXX: I think this just deletes Python objects, need to do more c "free" stuff here

    @staticmethod
    cdef GeoTessMetaData wrap(clib.GeoTessMetaData *cptr, owner=None):
        """ Wrap a C++ pointer with a pointer-less Python class.

        Deprecated.  Use `from_pointer` instead.
        """
        cdef GeoTessMetaData inst = GeoTessMetaData(raw=True)
        inst.thisptr = cptr
        if owner:
            inst.owner = owner

        return inst

    @staticmethod
    cdef GeoTessMetaData from_pointer(clib.GeoTessMetaData *cptr, owner=None):
        """ Initialize from a C++ GeoTessMetaData pointer.
        """
        cdef GeoTessMetaData wrapper = GeoTessMetaData.__new__(GeoTessMetaData, raw=True)
        wrapper.thisptr = cptr
        if owner:
            wrapper.owns_ptr = False

        return wrapper

    def setEarthShape(self, str earthShapeName):
        """ Specify the name of the ellipsoid that is to be used to convert between geocentric
        and geographic latitude and between depth and radius.

        This ellipsoid will be save in this GeoTessModel if it is written to file.
        The following EarthShapes are supported:

        SPHERE - Geocentric and geographic latitudes are identical and conversion between depth and
            radius assume the Earth is a sphere with constant radius of 6371 km.
        GRS80 - Conversion between geographic and geocentric latitudes, and between depth and
            radius are performed using the parameters of the GRS80 ellipsoid.
        GRS80_RCONST - Conversion between geographic and geocentric latitudes are performed using
            the parameters of the GRS80 ellipsoid. Conversions between depth and radius assume the
            Earth is a sphere with radius 6371.
        WGS84 - Conversion between geographic and geocentric latitudes, and between depth and radius
            are performed using the parameters of the WGS84 ellipsoid.
        WGS84_RCONST - Conversion between geographic and geocentric latitudes are performed
            using the parameters of the WGS84 ellipsoid. Conversions between depth and radius assume
            the Earth is a sphere with radius 6371.
        IERS2003 - Conversion between geographic and geocentric latitudes, and between depth and
            radius are performed using the parameters of the IERS2003 ellipsoid.
        IERS2003_RCONST - Conversion between geographic and geocentric latitudes are performed
            using the parameters of the IERS2003 ellipsoid. Conversions between depth and radius
            assume the Earth is a sphere with radius 6371.

        Parameters
        ----------
        earthShapeName : str
            the name of the ellipsoid that is to be used.

        """
        allowed_shapes = ('SPHERE', 'WGS84', 'WGS84_RCONST', 'GRS80', 'GRS80_RCONST',
                  'IERS2003', 'IERS2003_RCONST')
        if earthShapeName not in allowed_shapes:
            msg = "Unknown earth shape '{}'. Choose from {}"
            raise ValueError(msg.format(earthShapeName, allowed_shapes))
        self.thisptr.setEarthShape(earthShapeName)

    def setDescription(self, str dscr):
        """ Set the description of the model.

        Adds a newline as final character.

        Parameters
        ----------
        dscr : str
            the description of the model.
        """
        self.thisptr.setDescription(bytes(dscr, encoding='utf-8'))

    def getDescription(self):
        """ Retrieve the description of the model.

        Returns
        -------
        str
            the description of the model.
        """
        return self.thisptr.getDescription()

    def setLayerNames(self, str lyrNms):
        """ Specify the names of all the layers that comprise the model.

        This will determine the value of nLayers as well. The input lyrNms is a semicolon
        concatenation of all layer names (i.e. LAYERNAME1; LAYERNAME2; ...).
        Whitespaces will be removed.

        Parameters
        ----------
        lyrNms : str
            single string containing all the layer names separated by semi-colons
        """
        self.thisptr.setLayerNames(bytes(lyrNms, encoding='utf-8'))

    def setLayerTessIds(self, vector[int]& layrTsIds):
        """ LayerTessIds is a map from a layer index to a tessellation index.

        There is an element for each layer.

        Parameters
        ----------
        layrTsIds : iterable of int
            of length equal to the number of layers in the model.

        """
        # iterable of integers is automatically casted to vector of integers
        self.thisptr.setLayerTessIds(layrTsIds)

    def setAttributes(self, const string& nms, const string& unts):
        """ Specify the names and units of the attributes.

        Parameters
        ----------
        nms : str
            the names of the attributes, separated by a semicolon
        unts : str
            the units of the attributes, separated by a semicolon

        """
        self.thisptr.setAttributes(nms, unts)

    def setDataType(self, dt):
        """ Specify the type of the data that is stored in the model;

        Must be one of DOUBLE, FLOAT, LONG, INT, SHORTINT, BYTE.

        Parameters
        ----------
        dt : str
           the dataType to set.

        """
        allowed_dtypes = ('DOUBLE', 'FLOAT', 'LONG', 'INT', 'SHORTINT', 'BYTE')
        if dt not in allowed_dtypes:
            raise ValueError("DataType must be one of {}".format(allowed_dtypes))
        self.thisptr.setDataType(dt)

    def setModelSoftwareVersion(self, const string& swVersion):
        """ Set the name and version number of the software that generated the contents of this model.

        Parameters
        ----------
        swVersion : str

        """
        self.thisptr.setModelSoftwareVersion(swVersion)

    def getModelSoftwareVersion(self):
        """ Get the name and version of the software that generated the content of this model.

        Returns
        -------
        str
            the name and version of the software that generated this model.
        """
        self.thisptr.getModelSoftwareVersion()

    def setModelGenerationDate(self, const string& genDate):
        """ Set the date when this model was generated.

        This is not necessarily the same as the date when the file was copied or translated.

        Parameters
        ----------
        genDate	: str (free-form text date field)
        """
        self.thisptr.setModelGenerationDate(genDate)

    def toString(self):
        """ Returns a string representation of this meta data object
        """
        return self.thisptr.toString()

    def getAttributeNamesString(self):
        """ Retrieve the names of all the attributes assembled into a single, semi-colon separated string.

        Returns
        -------
        str
            the names of all the attributes assembled into a single, semi-colon separated string.
        """
        return self.thisptr.getAttributeNamesString()

    def getAttributeUnitsString(self):
        """ Retrieve the units of all the attributes assembled into a single, semi-colon separated string.

        Returns
        -------
        str
            the units of all the attributes assembled into a single, semi-colon separated string.
        """
        return self.thisptr.getAttributeUnitsString()

    def getLayerNamesString(self):
        """ Retrieve the names of all the layers assembled into a single, semi-colon separated string.

        Returns
        -------
        str
            the names of all the layers assembled into a single, semi-colon separated string.
        """
        return self.thisptr.getLayerNamesString()

    def getLayerTessIds(self):
        """ Retrieve a reference to layerTessIds

        An int[] with an entry for each layer specifying the index of the tessellation that supports that layer.

        Returns
        -------
        layerTessIds : list of int

        """
        # Use some internal NumPy C API calls to safely wrap the array pointer,
        # hopefully preventing memory leaks or segfaults.
        # following https://gist.github.com/aeberspaecher/1253698
        cdef const int *tess_ids = self.thisptr.getLayerTessIds()
        cdef np.npy_intp shape[1]
        cdef int nLayers = self.thisptr.getNLayers()
        shape[0] = <np.npy_intp> nLayers
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> tess_ids)
        np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_ARRAY_OWNDATA)

        return arr.tolist() # copies the data to a list.  XXX: this might leak memory.

    def getNLayers(self):
        """ Retrieve the number of layers represented in the model.

        Returns
        -------
        int
            number of layers represented in the model.
        """
        return self.thisptr.getNLayers()

    def getLayerName(self, const int &layerIndex):
        """ Retrieve the name of one of the layers supported by the model.

        Parameters
        ----------
        layerIndex : int
            the index of the layer

        Returns
        -------
        str
            the name of the layer
        """
        return self.thisptr.getLayerName(layerIndex)

    def getLayerIndex(self, str layerName):
        """ Retrieve the index of the layer that has the specified name, or -1.

        Parameters
        ----------
        layerName : str
            the name of the layer whose index is sought.

        Returns
        -------
        int
            the index of the layer that has the specified name, or -1.

        """
        # TODO: find out what "or -1" means here, and handle it in this python method
        return self.thisptr.getLayerIndex(bytes(layerName, encoding='utf-8'))

    def getModelFileFormat(self):
        # TODO: look up C++ docstring for this
        return self.thisptr.getModelFileFormat()

    def setModelFileFormat(self, version):
        # TODO: look up C++ docstring for this
        self.thisptr.setModelFileFormat(version)

    # def getEulerRotationAngles(self):
    #     """
    #     Retrieve the Euler Rotation Angles that are being used to rotate unit
    #     vectors from grid to model coordinates, in degrees. Returns null if no
    #     grid rotations are being applied. There are possibly two geographic coordinate
    #     systems at play:

    #     * Grid coordinates, where grid vertex 0 points to the north pole.
    #     * Model coordinates, where grid vertex 0 points to some other location,
    #       typically a station location.
    #
    #     Returns
    #     -------
    #     float
    #         euler rotation angles in degrees.

    #     """
    #     cdef double* rotation_angles_p = self.thisptr.getEulerRotationAngles()
    #     # rotaton_angles = deref(rotation_angles_p)

    #     return rotation_angles_p


cdef class EarthShape:
    """ Defines the ellipsoid that is to be used to convert between geocentric and
    geographic latitude and between depth and radius.

    EarthShape defines the ellipsoid that is to be used to convert between geocentric
    and geographic latitude and between depth and radius. The default is WGS84.

    The following EarthShapes are defined:

    Parameters
    ----------
    earthShape : str
        Define the shape of the Earth that is to be used to convert between geocentric and
        geographic latitude and between depth and radius.

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
        Convert geographic lat, lon into a geocentric unit vector.

        The x-component points toward lat,lon = 0, 0. The y-component points toward
        lat,lon = 0, 90. The z-component points toward north pole.

        Parameters
        ----------
        lat, lon : float

        Returns
        -------
        numpy.ndarray of floats

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
    """ Top level class that manages the GeoTessMetaData, GeoTessGrid and GeoTessData
    that comprise a 3D Earth model.

    GeoTessModel manages the grid and data that comprise a 3D Earth model. The Earth is assumed
    to be composed of a number of layers each of which spans the entire geographic extent of the
    Earth. It is assumed that layer boundaries do not fold back on themselves, i.e., along any
    radial profile through the model, each layer boundary is intersected exactly one time.
    Layers may have zero thickness over some or all of their geographic extent. Earth properties
    stored in the model are assumed to be continuous within a layer, both geographically and radially,
    but may be discontinuous across layer boundaries.

    A GeoTessModel is comprised of 3 major components:

    The model grid (geometry and topology) is managed by a GeoTessGrid object. The grid is made up
    of one or more 2D triangular tessellations of a unit sphere.

    The data are managed by a 2D array of Profile objects. A Profile is essentially a list of
    radii and Data objects distributed along a radial profile that spans a single layer at a
    single vertex of the 2D grid. The 2D Profile array has dimensions nVertices by nLayers.

    Important metadata about the model, such as the names of the major layers, the names of the
    data attributes stored in the model, etc., are managed by a GeoTessMetaData object.
    The term 'vertex' refers to a position in the 2D tessellation. They are 2D positions
    represented by unit vectors on a unit sphere. The term 'node' refers to a 1D position on a
    radial profile associated with a vertex and a layer in the model. Node indexes are unique
    only within a given profile (all profiles have a node with index 0 for example).
    The term 'point' refers to all the nodes in all the profiles of the model. There is only one
    'point' in the model with index 0. PointMap is introduced to manage all these different indexes.

    Parameterized constructor, specifying the grid and metadata for the model.

    The grid is constructed and the data structures are initialized based on information supplied in metadata.
    The data structures are not populated with any information however (all Profiles are null).
    The application should populate the new model's Profiles after this constructor completes.

    Before calling this constructor, the supplied MetaData object must be populated with required information
    by calling the following MetaData methods:

    setDescription()
    setLayerNames()
    setAttributes()
    setDataType()
    setLayerTessIds() (only required if grid has more than one multi-level tessellation)

    Parameters
    ----------
    gridFileName : str
        name of file from which to load the grid.
    metaData : geotess.lib.MetaData
        See Notes.

    Notes
    -----
    GeoTessModel accepts a grid file name and GeoTessMetaData instance.  The
    metadata is _copied_ into the GeoTessModel, so be warned that changes to
    it are _not_ reflected in the original instances.  This is done to
    simplify the life cycle of the underlying C++ memory, because GeoTessModel
    wants to assumes ownership of the provided C++ objects, including
    destruction.

    """
    # XXX: pointer ownership is an issue here.
    #   May have fixed some/all of it in the new .from_pointer staticmethod.
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

    # def loadModel(self, const string& inputFile, relGridFilePath=b""):
    def loadModel(self, str inputFile, relGridFilePath=b""):
        """ Read model data and grid from a file.

        Parameters
        ----------
        inputFile : str
            the path to the file that contains the model.
        relGridFilePath : str
            if the grid is stored in a separate file then relGridFilePath
            is the relative path from the directory where the model located
            to the directory where the grid is located. The default value for
            relGridFilePath is "" which indicates that the grid file resides
            in the same directory as the model file.

        Returns
        -------
        returns a pointer to this

        """
        # XXX: I don't know why this works, as its supposed to return a pointer to the loaded
        # model, not assign it to self.thisptr.
        # https://groups.google.com/forum/#!topic/cython-users/4ecKM-p8dPA
        if os.path.exists(inputFile):
            self.thisptr.loadModel(bytes(inputFile, encoding='utf-8'), relGridFilePath)
        else:
            raise exc.GeoTessFileError(f"File not found: {inputFile}.")

    def writeModel(self, const string& outputFile):
        """ Write the model to file.

        The data (radii and attribute values) are written to outputFile.
        If gridFileName is '*' or omitted then the grid information is written
        to the same file as the data. If gridFileName is something else, it should
        be the name of the file that contains the grid information (just the name;
        no path information). In the latter case, the gridFile referenced by
        gridFileName is not overwritten; all that happens is that the name of the
        grid file (with no path information) is stored in the data file.

        Parameters
        ----------
        outputFile : str
            name of the file to receive the model
        gridFileName : str
            name of file to receive the grid (no path info), or "*"

        """
        self.thisptr.writeModel(outputFile)

    def getConnectedVertices(self, int layerid):
        """ Function fo find which vertices are connected

        If a vertex is not connected, then it won't have a set profile.

        Paramters
        ---------
        layerID : int
            layer index

        Returns
        -------
        numpy.ndarray
            connected vertices at this layer

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

    def setPointDataSingleAttribute(self, pointIndex, attributeIndex, value):
        """
        For a given point index and attribute index, sets the value
        """
        ptMap = self.thisptr.getPointMap()
        geoData = ptMap.getPointData(pointIndex)
        geoData.setValue(attributeIndex, value)

    def getNearestPointIndex(self, float latitude, float longitude, float radius):
        """
        Warning! This does not always work. Layer definitions need to be included before it will work properly!
        This is also quite slow.

        Parameters
        ----------
        latitude : float
            floating point from -90 to 90
            Defines the latitude of the lookup point
        longitude : float
            floating point from -180 to 360
            Defines the longitude of the lookup point.
        radius : float
            floating point from 0 to ~6371 (earth's radius out from center')
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
        """ Returns a string with information about this model.
        """
        return self.thisptr.toString()

    def getEarthShape(self):
        """ Retrieve a reference to the ellipsoid that is stored in this GeoTessModel.

        This EarthShape object can be used to convert between geographic and geocentric latitude,
        and between radius and depth in the Earth.

        Returns
        -------
        Earthshape
            The EarthShape currently in use.
        """
        shp = EarthShape.wrap(&self.thisptr.getEarthShape(), owner=self)

        return shp

    def getMetaData(self):
        """ Return a reference to the GeoTessMetaData object associated with this model.

        The metadata object stores information about the models such as a description of the model,
        the layer names, attribute names, attribute units, the data type, etc.

        Returns
        -------
        GeoTessMetaData
            The metadata object.

        """
        md = GeoTessMetaData.wrap(&self.thisptr.getMetaData())
        md.owner = self

        return md

    def getNAttributes(self):
        """ Return the number of attributes that are associated with each node in the model.

        Returns
        -------
        int
            the number of attributes that are associated with each node in the model.

        """
        nattrib = self.thisptr.getNAttributes()

        return nattrib

    def getGrid(self):
        """ Return the current model's grid object.

        Returns
        -------
        GeoTessGrid
            Current model's grid object.

        """
        # XXX: I don't think this works.  It crashes the interpreter when the grid is deleted or
        # garbage collected. I need to fix pointer ownership or something.

        # cdef clib.GeoTessGrid *ptr = &self.thisptr.getGrid()
        # grid = lib.GeoTessGrid.from_pointer(ptr)

        # cdef GeoTessGrid grid = GeoTessGrid.wrap(&self.thisptr.getGrid())
        cdef GeoTessGrid grid = GeoTessGrid.from_pointer(&self.thisptr.getGrid())
        # grid.owner = self

        return grid

    def setProfile(self, int vertex, int layer, vector[float] &radii, vector[vector[float]] &values):
        # setProfile (int vertex, int layer, vector< float > &radii, vector< vector< T > > &values)
        # setProfile (const int &vertex, T *values, const int &nAttributes)
        """
        Set profile values at a vertex and layer.

        Parameters
        ----------
        vertex, layer : int
            vertex and layer number of the profile.
        radii : list
            Radius values of profile data.
        values : list of lists, or 2D numpy.ndarray
            List of corresponding attribute values at the provided radii.

        """
        # holycrap, vector[vector[...]] can just be a list of lists.  cython converts it.
        # I wonder if it can be a 2D NumPy array.  Yep!
        try:
            self.thisptr.setProfile(vertex, layer, radii, values)
        except Exception as e:
            # TODO: make this a more targeted exception
            msg = "Problem setting profile. Message: {}".format(e.message)
            raise Exception(msg)

    # def setProfileND(self, int vertex, int layer, radii, values):
    #     """
    #     Set profile values at a vertex and layer using ndarrays rather than c++ vector types

    #     Parameters
    #     ----------
    #     int vertex, layer
    #         vertex and layer indices of the profile
    #     radii : 1D ndarray
    #         ndarray radius values of the profile data
    #     values : 2D ndarray
    #         nradii x nattributes ndarray of attribute values at the provided radii

    #     Returns:
    #         1 on success
    #         -1 on values not being 2D ndarray
    #         -2 on errors packing ndarray in c++ vectors
    #         -3 on error setting profile values
    #     """
    #     import numpy as np
    #     cdef vector[float] cradii
    #     cdef vector[vector[float]] cvalues
    #     cdef vector[float] ctmp

    #     # Radii have to increase.
    #     # Put a check here to make sure the input radii and values ndarrays
    #     # are in increasing radius, that is radius outward from the center
    #     # of the earth
    #     if radii[1] < radii[0]:
    #         tmp = np.flipud(radii)
    #         radii = tmp.copy()
    #         tmp = np.flipud(values)
    #         values = tmp.copy()

    #     try:
    #         (nr, na) = values.shape
    #     except:
    #         print("Error in setProfileND: values must be nradii x nattributes ndarray")
    #         return -1
    #     try:
    #         cradii.reserve(nr)
    #         for ir, r in enumerate(radii):
    #             cradii.push_back(r)
    #             ctmp.clear()
    #             for ia, a in enumerate(values[ir]):
    #                 ctmp.push_back(a)
    #             cvalues.push_back(ctmp)
    #     except:
    #         print("Error in setProfileND: c++ vector fill error")
    #         return -2
    #     try:
    #         self.thisptr.setProfile(vertex, layer, cradii, cvalues)
    #         cradii.clear()
    #         cvalues.clear()
    #         return 1
    #     except:
    #         print("Error in setProfileND: c++ call failed.")
    #         return -3


    # XXX: I don't thing there's a GeoTessProfile.getTypeInt method in the library
    # def getProfileTypeInt(self, int vertex, int layer):
    #     """
    #     Given a vertex and layer, returns the profile type as an integer
    #     """
    #     A = self.thisptr.getProfile(vertex, layer)
    #     return A.getTypeInt()


    def getProfile(self, int vertex, int layer):
        """ Get a reference to the Profile object for the specified vertex and layer.

        Gets values in a profile given the vertex and layer.

        Parameters
        ----------
        vertex : int
            index of a vertex in the 2D grid
        layer : int
            index of one of the layers that comprise the model

        Returns
        -------
        radii : numpy.ndarray of floats [nradius x 1]
            Profile radius values for the profile.
        attributes : numpy.ndarray of floats [nradius x nattributes]
            Profile attribute values corresponding to each radius.

        """
        # returns nradius x 1 radius vector and nradius x nattributes attributes matrix
        # returns: a reference to a Profile object that contains the radii and Data values stored in profile[vertex][layer].
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
        """ Return the number of layers that comprise the model as an integer.
        """
        return self.thisptr.getNLayers()

    def getNVertices(self):
        """ Return number of vertices in the 2D geographic grid as an integer.
        """
        return self.thisptr.getNVertices()

    def getNPoints(self):
        """
        Retrieve the number of points in the model, including all nodes along all profiles
        at all grid vertices, as an integer.
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

        # pointA and pointB were specified as 1D typed memoryviews, so we can send the address to their first
        # element as the double pointer.
        # pointSpacing and radius are automatically converted to addresses by Cython because they
        # are numeric types and we specified them as typed in the calling signature.
        # The output of the C++ getWeights method is a map, which will automatically be converted to
        # a Python dict by Cython.

        cdef const clib.GeoTessInterpolatorType* interpolator
        cdef cmap[int, double] weights

        #
        if horizontalType not in ('LINEAR', 'NATURAL_NEIGHBOR'):
            msg = "horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'."
            raise ValueError(msg)

        # trouble with bytes vs str:
        # http://docs.cython.org/en/latest/src/tutorial/strings.html#accepting-strings-from-python-code
        # interpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        interpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType.encode('utf-8'))
        # cdef const clib.GeoTessInterpolatorType* interpolator = clib.GeoTessInterpolatorType.valueOf(bytes(horizontalType, encoding='utf-8'))

        #void getWeights(const double *pointA, const double *pointB, const double &pointSpacing,
        #                const double &radius, const GeoTessInterpolatorType &horizontalType, cmap[int, double]
        #                &weights) except +

        self.thisptr.getWeights(&pointA[0], &pointB[0], pointSpacing, radius,
                                deref(interpolator),
                                weights)

        return weights

    def getPointWeights(self, double lat, double lon, double radius, str horizontalType="LINEAR"):

        if horizontalType not in ('LINEAR', 'NATURAL_NEIGHBOR'):
            raise ValueError("horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'.")

        # cdef const clib.GeoTessInterpolatorType* interpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType.encode('utf-8'))
        cdef const clib.GeoTessInterpolatorType* interpolator = clib.GeoTessInterpolatorType.valueOf(bytes(horizontalType, encoding='utf-8'))
        pos = self.thisptr.getPosition(deref(interpolator))
        pos.set(lat, lon, radius)

        cdef cmap[int, double] weights

        pos.getWeights(weights,radius)
        return weights

    def getPointWeightsVector(self, double[:] v, double radius, str horizontalType="LINEAR"):

        if horizontalType not in ('LINEAR', 'NATURAL_NEIGHBOR'):
            raise ValueError("horizontalType must be either 'LINEAR' or 'NATURAL_NEIGHBOR'.")

        cdef const clib.GeoTessInterpolatorType* interpolator = clib.GeoTessInterpolatorType.valueOf(horizontalType)
        pos = self.thisptr.getPosition(deref(interpolator))
        pos.set(&v[0], radius)

        cdef cmap[int, double] weights

        pos.getWeights(weights,1.)
        return weights

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

    def setProfile(self, int vertex, int layer, vector[float] &radii, vector[vector[float]] &values):
        """
        Set profile values at a vertex and layer.
        This version works with c++ style vector types.
        Use setProfileND to push ndarrays instead.

        Parameters
        ----------
        vertex, layer : int
            vertex and layer number of the profile.
        radii : list
            Radius values of profile data.
        values : list of lists
            List of corresponding attribute values at the provided radii.

        Returns:
            -1 on failure

        """

        try:
            self.thisptr.setProfile(vertex, layer, radii, values)
        except:
            raise ValueError('setProfile failed')

#     def setProfileND(self, int vertex, int layer, radii, values):
#         """
#         Set profile values at a vertex and layer using ndarrays rather than c++ vector types

#         Parameters
#         ----------
#         int vertex, layer
#             vertex and layer indices of the profile
#         radii : 1D ndarray
#             ndarray radius values of the profile data
#         values : 2D ndarray
#             nradii x nattributes ndarray of attribute values at the provided radii

#         Returns:
#             1 on success
#             -1 on values not being 2D ndarray
#             -2 on errors packing ndarray in c++ vectors
#             -3 on error setting profile values
#         """
#         import numpy as np
#         cdef vector[float] cradii
#         cdef vector[vector[float]] cvalues
#         cdef vector[float] ctmp

#         # Radii have to increase.
#         # Put a check here to make sure the input radii and values ndarrays
#         # are in increasing radius, that is radius outward from the center
#         # of the earth
#         if radii[1] < radii[0]:
#             tmp = np.flipud(radii)
#             radii = tmp.copy()
#             tmp = np.flipud(values)
#             values = tmp.copy()

#         try:
#             (nr, na) = values.shape
#         except:
#             print("Error in setProfileND: values must be nradii x nattributes ndarray")
#             return -1
#         try:
#             cradii.reserve(nr)
#             for ir, r in enumerate(radii):
#                 cradii.push_back(r)
#                 ctmp.clear()
#                 for ia, a in enumerate(values[ir]):
#                     ctmp.push_back(a)
#                 cvalues.push_back(ctmp)
#         except:
#             print("Error in setProfileND: c++ vector fill error")
#             return -2
#         try:
#             self.thisptr.setProfile(vertex, layer, cradii, cvalues)
#             cradii.clear()
#             cvalues.clear()
#             return 1
#         except:
#             print("Error in setProfileND: c++ call failed.")
#             return -3


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

    def getValueFloat(self, int pointIndex, int attributeIndex):
        """ Return the value of the attribute at the specified pointIndex, attributeIndex,
        cast to a float if necessary.

        Parameters
        ----------
        pointIndex : int
        attributeIndex : int
            the attributeIndex

        Returns
        -------
        float
            the value of the specifed attribute, cast to float if necessary

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
        # GeoTessPosition* getPosition(const GeoTessInterpolatorType& horizontalType, const GeoTessInterpolatorType& radialType)
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
            depth from surface of ellipsoid. [km] I think.
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
        cdef vector[vector[double]] attributes
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
        pos.getBorehole(dz, computeDepthFlag, layers, radii, attributes)
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

    #def getVertexLayerPosition(self, float lat, float lon, float depth, horizontalType="LINEAR", radialType="LINEAR"):
    #    """
    #    (Placeholder method)
    #    Given coordinates in latitude, longitude, and depth, finds the vertex and layer indices
    #    """
    #    print("Error, this method has not been built yet.")
    #    return

    # Should get this from GeoTessModelUtils
    # Needs an update based on updated getGeographicLocationAttribute() method
    def makeDepthMap(self, float depth, int attribute, int layer, float dLon = 8.0,
                     float dLat=8.0, float minlon=0, float maxlon=360, float minlat=-90, float maxlat=90,
                     horizontalType="LINEAR", radialType="LINEAR"):
        """
        Extracts values for a map at constant depth.
        The output from this can be used to make a map with other software, such as matplotlib
        Required positional arguments: depth, attribute index.
        Optional arguments:
            dLon: gridding step in longitude
            dLat: gridding step in latitude
            minlon: minimum longitude in degrees
            maxlon: maximum longitude in degrees
            minlat: minimum latitude in degrees
            maxlat: maximum latitude in degrees
        relies on numpy as np

        """
        import numpy as np
        lons = np.arange(minlon, maxlon, dLon)
        lats = np.arange(minlat, maxlat, dLat)
        outData = np.zeros((len(lons), len(lats)))
        for ilon, lon in enumerate(lons):
            for ilat, lat in enumerate(lats):
                radius = self.positionGetRadius(lat, lon, depth, horizontalType=horizontalType, radialType=radialType)
                outData[ilon, ilat] = self.getGeographicLocationAttribute(lat, lon, radius, attribute, layer, horizontalType=horizontalType, radialType=radialType)

        return lons, lats, outData

    # Should get this from GeoTessModelUtils
    def make1DProfile(self, float lat, float lon, int attribute, float mindepth=0, float maxdepth=6371, float dz = 1, horizontalType="LINEAR", radialType="LINEAR"):
        """
        Extracts values as a 1-dimensional array of depth and attribute
        Returns numpy arrays of depth and value
        optional parameters:
            mindepth: minimum depth (km)
            maxdepth: maximum depth (km)
            dz: sampling in depth (km)
        """
        depths = np.arange(mindepth, maxdepth, dz)
        outData = np.zeros((len(depths),))
        for idepth, depth in enumerate(depths):
            radius = self.positionGetRadius(lat, lon, depth, horizontalType=horizontalType, radialType=radialType)
            layer = self.positionGetLayer(lat, lon, depth, horizontalType=horizontalType, radialType=radialType)
            outData[idepth] = self.getGeographicLocationAttribute(lat, lon, radius, attribute, layer, horizontalType=horizontalType, radialType=radialType)

        return depths, outData

    def convertToNPArray(self):
        """
        Extracts from geotess object to a set of 3 location vectors and an attribute matrix
        returns longitude vector, latitude vector, radius vector, and data matrix
        """
        import numpy as np
        grid = self.getGrid()
        ellipsoid = self.getEarthShape()

        npts = 0
        for layer in range(self.getNLayers()):
            for vtx in range(self.getNVertices()):
                #print(vtx, layer)
                rads, att = self.getProfile(vtx, layer)
                #print(len(rads))
                npts += len(rads)

        nparams = self.getNAttributes()

        lonsOut = np.zeros((npts,))
        latsOut = np.zeros((npts,))
        radsOut = np.zeros((npts,))
        dataOut = np.zeros((npts, nparams))
        idx = 0
        for layer in range(self.getNLayers()):
            for vtx in range(self.getNVertices()):
                vertex = grid.getVertex(vtx)
                lat = ellipsoid.getLatDegrees(vertex)
                lon = ellipsoid.getLonDegrees(vertex)
                rads, att = self.getProfile(vtx, layer)
                # Need proper err
                for irad, rad in enumerate(rads):
                    lonsOut[idx] = lon
                    latsOut[idx] = lat
                    radsOut[idx] = rad
                    if att is not None:
                        for iat in range(nparams):
                            dataOut[idx, iat] = att[irad, iat]
                    idx += 1
        return lonsOut, latsOut, radsOut, dataOut


cdef class AK135Model:
    # this class is simple enough & has only a default/empty constructor, so it
    # may be able to use the simplified wrapping approach
    # https://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#simplified-wrapping-with-default-constructor
    # "if the class has only one constructor and it is a nullary one, it’s not necessary to declare it."
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
        if modelInputFile is None:
            self.thisampptr = new clib.GeoTessModelAmplitude()
        else:
            self.thisampptr = new clib.GeoTessModelAmplitude(modelInputFile)

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
