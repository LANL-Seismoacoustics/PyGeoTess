#distutils: language = c++

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

