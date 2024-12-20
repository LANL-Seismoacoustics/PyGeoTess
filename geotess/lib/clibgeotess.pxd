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
from libcpp.map cimport map as cmap
from libcpp.set cimport set


cdef extern from "geotesscpp/GeoTessModelUtils.h" namespace "geotess":
    cdef cppclass GeoTessModelUtils:
        GeoTessModelUtils() except +
        int updatePointsPerLayer(GeoTessPosition& pos, int firstLayer, int lastLayer, double maxSpacing, vector[int]& pointsPerLayer)
        # These need to be refactored to replace bools with integer flags
        #string getBoreholeString(GeoTessModel& pos, double lat, double lon)
        #string getBoreholeString(GeoTessPosition& pos, double maxSpacing, int firstLayer, int lastLayer, bool convertToDepth, bool reciprocal, vector[int]& attributes)
        # The rest of the functions use 2D vectors of vectors. I need to make flattened versions for easier passing


cdef extern from "geotesscpp/GeoTessUtils.h" namespace "geotess":
    cdef cppclass GeoTessUtils:
        GeoTessUtils() except +
        # a lot of these methods are static, so we use @staticmethod
        # https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#static-member-method
        # This makes them like functions within a "GeoTessUtils" Python module
        # instead of methods on a class instance.
        # try to match common C++ exceptions to Python ones: https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        @staticmethod
        double getLatDegrees(const double *const v)
        @staticmethod
        double getLonDegrees(const double *const v)
        @staticmethod
        double* getVectorDegrees(const double &lat, const double &lon, double *v)
        @staticmethod
        double getEarthRadius(const double *const v)

cdef extern from "geotesscpp/GeoTessGrid.h" namespace "geotess":
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

cdef extern from "geotesscpp/GeoTessMetaData.h" namespace "geotess":
    cdef cppclass GeoTessMetaData:
        GeoTessMetaData() except +
        GeoTessMetaData(const GeoTessMetaData &md)
        void setEarthShape(const string& earthShapeName)
        #const string& getEarthShapeName() const
        EarthShape& getEarthShape()
        void setDescription(const string& dscr)
        const string& getDescription() const
        void setLayerNames(const string& lyrNms)
        void setLayerTessIds(vector[int]& layrTsIds)
        # apparently, vector<int> in c++ is vector[int] in Cython
        void setAttributes(const string& nms, const string& unts)
        void setDataType(const string& dt)
        void setModelSoftwareVersion(const string& swVersion)
        string getModelSoftwareVersion() const
        void setModelGenerationDate(const string& genDate)
        GeoTessMetaData* copy()
        string toString() const
        string getAttributeNamesString() const
        string getAttributeUnitsString() const
        string getLayerNamesString()
        const int* getLayerTessIds() const
        int getLayerIndex(const string& layerName) const
        int getNLayers() const
        string getLayerName(const int &layerIndex)
        int getModelFileFormat() const
        void setModelFileFormat(int version) const

cdef extern from "geotesscpp/EarthShape.h" namespace "geotess":
    cdef cppclass EarthShape:
        EarthShape(const string &earthShape) except +
        double getLonDegrees(const double *const v)
        double getLatDegrees(const double *const v)
        void getVectorDegrees(const double &lat, const double &lon, double *v)

cdef extern from "geotesscpp/GeoTessModel.h" namespace "geotess":
    cdef cppclass GeoTessModel:
        GeoTessModel() except +
        GeoTessModel(const string &gridFileName, GeoTessMetaData *metaData) except +
        # methods with default values can't be declared as such here.  they are
        # to be handled in the pyx file.
        GeoTessModel* loadModel(const string& inputFile, const string& relGridFilePath)
        void writeModel(const string &outputFile) except +
        string toString()
        GeoTessPointMap* getPointMap()
        EarthShape& getEarthShape()
        GeoTessMetaData& getMetaData()
        GeoTessGrid& getGrid()
        void setProfile(int vertex, int layer, vector[float] &radii, vector[vector[float]] &values)
        #void setProfile(int vertex, int layer, GeoTessProfile* profile)
        GeoTessProfile* getProfile(int vertex, int layer)
        #int getProfile(int vertex, int layer)
        int getNLayers() const
        int getNVertices() const
        int getNPoints() const
        int getNRadii(int vertexId, int layerId)
        void getWeights(const double *pointA, const double *pointB, const double &pointSpacing, const double &radius, const GeoTessInterpolatorType &horizontalType, cmap[int, double] &weights) except +
        float getValueFloat(int pointIndex, int attributeIndex)
        float getRadius(int pointIndex) const
        float getDepth(int pointIndex) const
        set[int]& getConnectedVertices(int layerIndex)
        GeoTessPosition* getPosition()
        GeoTessPosition* getPosition(const GeoTessInterpolatorType& horizontalType)
        GeoTessPosition* getPosition(const GeoTessInterpolatorType& horizontalType, const GeoTessInterpolatorType& radialType)

cdef extern from "geotesscpp/AK135Model.h" namespace "geotess":
    cdef cppclass AK135Model:
        AK135Model() except +
        void getLayerProfile(const double &lat, const double &lon, const int &layer, vector[float] &r, vector[vector[float]] &nodeData)

cdef extern from "geotesscpp/GeoTessData.h" namespace "geotess":
    cdef cppclass GeoTessData:
        GeoTessData() except +
        double getDouble(int attributeIndex) const
        float getFloat(int attributeIndex) const
        void setValue(int attributeIndex, double v)
        int size() const

cdef extern from "geotesscpp/GeoTessProfile.h" namespace "geotess":
    cdef cppclass GeoTessProfile:
        GeoTessProfile() except +
        int getNRadii() const
        int getNData() const
        float * getRadii() const
        GeoTessData* getData(int i)
        int getTypeInt()

cdef extern from "geotesscpp/GeoTessPointMap.h" namespace "geotess":
    cdef cppclass GeoTessPointMap:
        GeoTessPointMap() except +
        int size() const
        int getVertexIndex(int pointIndex) const
        int getTessId(int pointIndex) const
        int getLayerIndex(int pointIndex) const
        int getNodeIndex(int pointIndex) const
        int getPointIndex(int vertex, int layer, int node) const
        int getPointIndexLast(int vertex, int layer) const
        int getPointIndexFirst(int vertex, int layer) const
        GeoTessData* getPointData(int pointIndex)
        void setPointData(int pointIndex, GeoTessData* data)
        double getPointValue(int pointIndex, int attributeIndex)
        double getDistance3D(int pointIndex1, int pointIndex2)
        void getPointVector(int pointIndex, double* v)
        string getPointLatLonString(int pointIndex)
        string toString(int pointIndex)

# GeoTessPosition objects are called from within the GeoTessModel object
# Several methods are overloaded, that is flexible on their input parameters, which c++ is ok with, but python freaks out over
# The setters and getters only operate within GeoTessModel methods and therefore are not sticky in the calling python
cdef extern from "geotesscpp/GeoTessPosition.h" namespace "geotess":
    cdef cppclass GeoTessPosition:
        GeoTessPosition() except +
        GeoTessPosition* getGeoTessPosition(GeoTessModel* model)
        GeoTessPosition* getGeoTessPosition(GeoTessModel* model, const GeoTessInterpolatorType& horizontalType)
        GeoTessPosition* getGeoTessPosition(GeoTessModel* model, const GeoTessInterpolatorType& horizontalType, const GeoTessInterpolatorType& radialType);
        GeoTessInterpolatorType& getInterpolatorType() const
        void setModel(GeoTessModel* newModel) const
        void set(double lat, double lon, double depth)
        void set(int layid, double lat, double lon, double depth)
        #additional set method in GeoTessPosition
        # void set(const double *const v, double radius)
        void set(const double* const, const double&)
        #additional set method in GeoTessPosition
        void setTop(int layid, const double* const uVector)
        void setBottom(int layid, const double* const uVector)
        double getValue(int attribute)
        double getRadiusBottom(int layer)
        double getRadiusTop(int layer)
        double getRadiusBottom()
        double getRadiusTop()
        double getDepth()
        void setRadius(int layer, double r) const
        int getLayerId(double rad)
        void setDepth(int layer, double depth)
        void setDepth(double depth)
        void setTop(int layid)
        void setBottom(int layid)
        double getEarthRadius() const
        double* getVector()
        int getTriangle()
        int getNVertices()
        int getIndexOfClosestVertex() const
        const double* getClosestVertex() const
        string toString() const
        # This declaration of getVertex is unclear. It's help is:
        # Return the index of one of the vertices used to interpolate data.
        int getVertex(int index)
        ############################
        void setMaxTessLevel(int layid, int maxTess)
        int getMaxTessLevel(int layid)
        int getTessLevel() const
        int getTessLevel(const int& tId)
        int getTessID()
        double getRadius()
        double getLayerThickness()
        double getLayerThickness(int layid)
        double getDepthBottom(int layid)
        double getDepthTop(int layid)
        double getDepthBottom()
        double getDepthTop()
        int getLayerId()
        void getBorehole(double rSpacing, int computeDepth, vector[int]& layers, vector[double]& radii, vector[vector[double]]& attributes)
        string radialInterpolatorToString()
        void getWeights(cmap[int, double]&, double) except +

cdef extern from "geotesscpp/GeoTessInterpolatorType.h" namespace "geotess":
    cdef cppclass GeoTessInterpolatorType:
        @staticmethod
        GeoTessInterpolatorType* valueOf(const string &s)
        int size() const
        # I can't seem to access public const members in Cython
        # https://stackoverflow.com/a/46998685/745557
        # const GeoTessInterpolatorType LINEAR
        # const GeoTessInterpolatorType NATURAL_NEIGHBOR
        # const GeoTessInterpolatorType CUBIC_SPLINE

# This is protected, so I can't make new enum types.
# Rather, these need to be called from something on top of it, such as GeoTessInterpolatorType
#cdef extern from "GeoTessEnumType.h" namespace "geotess":
#    cdef cppclass GeoTessEnumType:
#        GeoTessEnumType() except+
#        string toString() const
#        string name() const
#        int ordinal() const

# GeoTessModelAmplitude is a subclass of GeoTessModel.  Hence the longer name.
# https://altugkarakurt.github.io/how-to-wrap-polymorphic-cpp-classes-with-cython
cdef extern from "geotesscpp/GeoTessModelAmplitude.h" namespace "geotess":
    cdef cppclass GeoTessModelAmplitude(GeoTessModel):
        GeoTessModelAmplitude() except +
        GeoTessModelAmplitude(const string& modelInputFile);
        float getSiteTrans(const string& station, const string& channel, const string& band)
