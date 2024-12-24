#distutils: language = c++
#cython: embedsignature=True
#cython: language_level=3
#cython: c_string_type=unicode
#cython: c_string_encoding=utf-8

import numpy as np
cimport numpy as np

np.import_array()

cimport geotess.lib.geotesscpp as clib

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



