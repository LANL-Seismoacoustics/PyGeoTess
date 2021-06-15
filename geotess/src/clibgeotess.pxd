#distutils: language = c++
"""
This module pulls GeoTess c++ functionality into Cython (not in Python yet).

A pxd file is just a header file, mirroring that of c++.  It's necessary
because Cython doesn't have a C header parser.

For simplicity, we pull from all of GeoTess into this one place, and only the
desired classes, methods, and functions.  Declarations are mostly unchanged
from the original.

"""
# libcpp is a Cython thing
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "GeoTessUtils.h" namespace "geotess":
    cdef cppclass GeoTessUtils:
        GeoTessUtils() except +
        # a lot of these methods are static, so we use @staticmethod
        # https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#static-member-method
        # This makes them like functions within a "GeoTessUtils" Python module
        # instead of methods on a class instance.
        @staticmethod
        double getLatDegrees(const double *const v)
        @staticmethod
        double getLonDegrees(const double *const v)
        @staticmethod
        double* getVectorDegrees(const double &lat, const double &lon, double *v)
        @staticmethod
        double getEarthRadius(const double *const v)

cdef extern from "GeoTessGrid.h" namespace "geotess":
    cdef cppclass GeoTessGrid:
        GeoTessGrid() except +
        GeoTessGrid(GeoTessGrid &other) except +
        GeoTessGrid* loadGrid(const string& inputFile)
        void writeGrid(const string& fileName)
        int getNLevels() const
        int getNLevels(int tessellation) 
        int getNTriangles()
        int getNTriangles(int tessellation, int level) const
        int getNTessellations()
        int getNVertices() const
        string toString()
        const double* getVertex(int vertex) const
        const vector[int] getVertexTriangles(const int &tessId, const int &level, const int &vertex) const
        const int* getTriangleVertexIndexes(int triangleIndex) const
        int getFirstTriangle(int tessellation, int level) const
        int getLastTriangle(int tessellation, int level) const
        int getVertexIndex(int triangle, int corner) const
        # had to remove a "const" from the def
        # Cython can't use 'const' in all the same places as C++
        # http://stackoverflow.com/questions/23873652/how-to-use-const-in-cython
        double *const * getVertices() const

cdef extern from "GeoTessMetaData.h" namespace "geotess":
    cdef cppclass GeoTessMetaData:
        GeoTessMetaData() except +
        GeoTessMetaData(const GeoTessMetaData &md)
        void setEarthShape(const string& earthShapeName)
        void setDescription(const string& dscr)
        const string& getDescription() const
        void setLayerNames(const string& lyrNms)
        void setLayerTessIds(vector[int]& layrTsIds)
        # apparently, vector<int> in c++ is vector[int] in Cython
        void setAttributes(const string& nms, const string& unts)
        void setDataType(const string& dt)
        void setModelSoftwareVersion(const string& swVersion)
        void setModelGenerationDate(const string& genDate)
        GeoTessMetaData* copy()
        string toString() const
        string getAttributeNamesString() const
        string getAttributeUnitsString() const
        string getLayerNamesString()
        const int* getLayerTessIds() const
        int getNLayers() const
        string getLayerName(const int &layerIndex) 	

cdef extern from "EarthShape.h" namespace "geotess":
    cdef cppclass EarthShape:
        EarthShape(const string &earthShape) except +
        double getLonDegrees(const double *const v)
        double getLatDegrees(const double *const v)
        void getVectorDegrees(const double &lat, const double &lon, double *v)

cdef extern from "GeoTessModel.h" namespace "geotess":
    cdef cppclass GeoTessModel:
        GeoTessModel() except +
        GeoTessModel(const string &gridFileName, GeoTessMetaData *metaData) except +
        # methods with default values can't be declared as such here.  they are
        # to be handled in the pyx file.
        GeoTessModel* loadModel(const string& inputFile, const string& relGridFilePath)
        void writeModel(const string &outputFile) except +
        string toString()
        EarthShape& getEarthShape()
        GeoTessMetaData& getMetaData()
        GeoTessGrid& getGrid()
        void setProfile(int vertex, int layer, vector[float] &radii, vector[vector[float]] &values)
        GeoTessProfile* getProfile(int vertex, int layer)
        int getNLayers() const
        int getNVertices() const

cdef extern from "AK135Model.h" namespace "geotess":
    cdef cppclass AK135Model:
        AK135Model() except +
        void getLayerProfile(const double &lat, const double &lon, const int &layer, vector[float] &r, vector[vector[float]] &nodeData)

cdef extern from "GeoTessProfile.h" namespace "geotess":
    cdef cppclass GeoTessProfile:
        GeoTessProfile() except +
