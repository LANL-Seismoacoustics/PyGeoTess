import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport malloc, free

cimport geotess.lib.geotesscpp as clib

# def ndarray_to_double_ptr(array2D):
#     """ Get a double** array (1D) of pointers for a 2D NumPy array. """
#     # from: https://abstract-theory.github.io/using-ctypes-with-double-pointers.html
#     # array2D.ctypes.data is the memory address (array offset)
#     return array2D.ctypes.data + array2D.strides[0]*np.arange(array2D.shape[0], dtype=np.uintp)

# cdef double** ndarray_to_double_ptr(np.ndarray[double, ndim=2] arr):
#     cdef:
#         int i
#         double **ptr
# 
#     # Get the shape of the NumPy array
#     rows, cols = arr.shape
# 
#     # Allocate memory for the double** pointer
#     ptr = <double **> malloc(rows * sizeof(double *))
# 
#     # Iterate over the rows of the NumPy array
#     for i in range(rows):
#         # Get a pointer to the current row
#         ptr[i] = &arr[i, 0]
# 
#     return ptr

# cdef double_ptr_to_ndarray(double **ptr, int rows, int cols):
#     cdef:
#         np.ndarray[double, ndim=2] arr
# 
#     # Create a NumPy ndarray object that owns the data
#     arr = np.PyArray_NewFromData(
#         np.NPY_DOUBLE,  # dtype
#         2,  # ndim
#         [rows, cols],  # shape
#         np.NPY_ARRAY_OWNDATA,  # flags
#         ptr,  # data
#         NULL,  # strides
#         NULL  # descr
#     )
# 
#     return arr

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

    @staticmethod
    def getGreatCircle(double[:] v0, double azimuth):
        """
        This method returns a great circle that is defined by an initial point and an azimuth.

        A great circle is defined by two unit vectors that are 90 degrees
        apart. A great circle is stored in a double[2][3] array, which is the
        structure returned by this method. A great circle can be passed to the
        method getGreatCirclePoint() to retrieve a unit vector that is on the
        great circle and located some distance from the first point of the
        great circle.

        This method returns a great circle that is defined by an initial point and an azimuth.

        Parameters
        ----------
        v0 : 3-element array of floats
            a unit vector that will be the first point on the great circle.
        azimuth : float
            a direction, in radians, in which to move relative to v in order to define the great circle

        Returns
        -------
        gc : a 2 x 3 array specifying two unit vectors. The first one is a clone of
            unit vector v0 passed as first argument to this method. The second is
            located 90 degrees away from v0.

        Raises
        ------
        GeoTessException
            if v is located at north or south pole.

        """
        cdef:
            int i
            double **ptr

        # initialize the output array (and it's C memory, which is contiguous linear)
        gc = np.empty((2, 3), dtype=np.double, order='C')
        cdef double[:, :] gc_memview = gc

        # Get the shape of the NumPy array
        rows, cols = gc.shape

        # getGreatCircle describes array memory differently than NumPy
        # Translate NumPy contiguous memory into an array of pointers for getGreatCircle .
        ptr = <double **> malloc(rows * sizeof(double *))

        # Iterate over the rows of the NumPy array
        for i in range(rows):
            # Get a pointer to the current row
            ptr[i] = &gc_memview[i, 0]

        # static void getGreatCircle(const double* const v, double azimuth, double** const gc);
        clib.GeoTessUtils.getGreatCircle(&v0[0], azimuth, ptr)

        # NumPy memory has been modified via the array of pointers, so we're done with it now
        free(ptr)

        return gc


    @staticmethod
    def getGreatCirclePoint(double[:, :] greatCircle, double distance):
        """
	A great circle is defined by two unit vectors that are 90 degrees apart.
	A great circle is stored in a double[2][3] array and one can be obtained
	by calling one of the getGreatCircle() methods.

	In this method, a great circle and a distance are specified and a point
	is returned which is on the great circle path and is the specified
	distance away from the first point of the great circle.
	
	Parameters
        ----------
        greatCircle : array-like 2x3 of floats 
            a great circle structure
        distance : float
            distance in radians from first point of great circle

        Returns
        -------
        unit vector of point which is on great circle and located specified
        distance away from first point of great circle.

        """
        cdef double **ptr

        # get a double** pointer array for the greatCircle input array
        cdef int rows = greatCircle.shape[0]
        ptr = <double **> malloc(rows * sizeof(double *))
        for i in range(rows):
            ptr[i] = &greatCircle[i, 0]

        v = np.empty(3, dtype=np.double)
        cdef double[:] v_memview = v

	# getGreatCirclePoint(double const* const * const greatCircle, double distance, double* const v)
        clib.GeoTessUtils.getGreatCirclePoint(ptr, distance, &v_memview[0])
        free(ptr)

        # Use the NumPy C API to get a pythonland NumPy array using data from 
        # double* v = clib.GeoTessUtils.getGreatCirclePoint(...)
        # cdef np.ndarray[np.double, ndim=1] out = np.PyArray_SimpleNewFromData(1, &dim, np.NPY_DOUBLE, (void *)v)
        # the lifetime of the data is tied to the NumPy array
        # np.PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

        return v
