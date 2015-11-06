"""
This module pulls GeoTess c++ functionality into Cython (not in Python yet).

A pxd file is just a header file, mirroring that of c++.  It's necessary
because Cython doesn't have a C header parser.

We pull from all of GeoTess into this one place, and only the desired classes,
methods, and functions.  Declarations are mostly unchanged from the original.

"""
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "GeoTessGrid.h" namespace "geotess":
    cdef cppclass GeoTessGrid:
        GeoTessGrid() except +
        GeoTessGrid(GeoTessGrid &other) except +
        # string value inputFile is turned into a pointer, that can't be used to
        # modify the thing it points to, and returns a pointer to a GeoTessGrid.
        GeoTessGrid* loadGrid(const string& inputFile)
        void writeGrid(const string& fileName)
        int getNLevels()
        int getNTriangles()
        int getNTessellations()
        string toString()
        const double* getVertex(int vertex) const

cdef extern from "GeoTessMetaData.h" namespace "geotess":
    cdef cppclass GeoTessMetaData:
        GeoTessMetaData() except +
        GeoTessMetaData(const GeoTessMetaData &md)
        void setEarthShape(const string& earthShapeName)
        void setDescription(const string& dscr)
        void setLayerNames(const string& lyrNms)
        void setLayerTessIds(vector[int]& layrTsIds)
        # apparently, vector<int> in c++ is vector[int] in Cython
        void setAttributes(const string& nms, const string& unts)
        void setDataType(const string& dt)
        void setModelSoftwareVersion(const string& swVersion)
        void setModelGenerationDate(const string& genDate)
        string toString() const

cdef extern from "EarthShape.h" namespace "geotess":
    cdef cppclass EarthShape:
        EarthShape(const string &earthShape) except +
        double getLonDegrees(const double *const v)
        double getLatDegrees(const double *const v)
        void getVectorDegrees(const double &lat, const double &lon, double *v)

cdef extern from "GeoTessModel.h" namespace "geotess":
    cdef cppclass GeoTessModel:
        GeoTessModel() except +
        GeoTessModel(GeoTessGrid *grid, GeoTessMetaData *metaData) except +
        # methods with default values can't be declared as such here.  they are
        # to be handled in the pyx file.
        GeoTessModel* loadModel(const string& inputFile, const string& relGridFilePath)
        void writeModel(const string &outputFile)
        string toString()
        EarthShape& getEarthShape()
