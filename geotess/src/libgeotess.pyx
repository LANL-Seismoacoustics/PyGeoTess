"""
This module exposes Cython GeoTess functionality from the pxd file into Python.

The class definitions here are Python-visible, and are simply wrappers that 
forward the Python-exposed methods directly to their Cython-exposed c++
counterparts, which have been exposed in the imported pxd file.

Using both a pxd and a pyx file is done, partly, so that we can keep the
exposed c++ GeoTess functionality together in one namespace using "cimport",
and we can name the classes exposed to Python the same as those in the
GeoTess c++.

"""
from libcpp.string cimport string

cimport clibgeotess as clib

cdef class GeoTessGrid:
    cdef clib.GeoTessGrid *thisptr

    def __cinit__(self):
        self.thisptr = new clib.GeoTessGrid()

    def __dealloc__(self):
        del self.thisptr

    def loadGrid(self, const string& inputFile):
        self.thisptr.loadGrid(inputFile)

    def writeGrid(self, const string& fileName):
        self.thisptr.writeGrid(fileName)

    def getNLevels(self):
        return self.thisptr.getNLevels()
 
    def getNTriangles(self):
        return self.thisptr.getNTriangles()

    def getNTessellations(self):
        return self.thisptr.getNTessellations()

    def toString(self):
        # XXX: doesn't work, don't know why
        return self.thisptr.toString()


cdef class GeoTessModel:
    cdef clib.GeoTessModel *thiptr

    def __cinit__(self):
        self.thisptr = new clib.GeoTessModel()

    def __dealloc__(self):
        del self.thisptr

    def loadModel(self, const string &inputFile, const string &relGridFilePath=""):
        self.thisptr.loadModel(inputFile, relGridFilePath)

    def writeModel(const string &outputFile):
        self.thisptr.writeModel(outputFile)
