"""
Home of the GeoTess Model.

Model is the main user-facing GeoTess class.  GeoTess Grids are build by 
geotessbuilder, and rarely (never) directly manipulated.  GeoTess MetaData
instances are mostly hidden within GeoTess Models.  The Model is the access
layer and manager of each of these other classes.

"""
from geotess.utils import Layer, Attribute
from geotess.grid import Grid
from geotess.libgeotess import GeoTessModel

class Model(object):
        """
        Initialize a Model using a grid file and metadata.

        Parameters
        ----------
        gridfile : str
            Full path to an existing GeoTess grid file.  If it's an ascii grid,
            the extension should be ".ascii".
        layers : list of geotess.util.Layer tuples
            Layer[0] is the layer name string
            Layer[1] is the layer tessellation id integer
        attributes : list of geotess.util.Attribute tuples
            Attribute[0] is the attribute name string
            Attribute[1] is the attribute unit string
        earth_shape : str {"sphere", "grs80", "grs80_rconst", "wgs84",
                           "wgs84_rconst"}
            Defines the ellipsoid that is to be used to convert between geocentric
            and geographic latitude and between depth and radius. The default is
            wgs84.

            * sphere - Geocentric and geographic latitudes are identical and
            conversion between depth and radius assume the Earth is a sphere
            with constant radius of 6371 km.
            * grs80 - Conversion between geographic and geocentric latitudes,
            and between depth and radius are performed using the parameters of
            the GRS80 ellipsoid.
            * grs80_rconst - Conversion between geographic and geocentric
            latitudes are performed using the parameters of the GRS80
            ellipsoid. Conversions between depth and radius assume the Earth is
            a sphere with radius 6371.
            * wgs84 - Conversion between geographic and geocentric latitudes,
            and between depth and radius are performed using the parameters of
            the WGS84 ellipsoid.
            * wgs84_rconst - Conversion between geographic and geocentric
            latitudes are performed using the parameters of the WGS84
            ellipsoid. Conversions between depth and radius assume the Earth is
            a sphere with radius 6371.
            * iers2003 - Conversion between geographic and geocentric
            latitudes, and between depth and radius are performed using the
            parameters of the IERS2003 ellipsoid.
            * iers2003_rconst - Conversion between geographic and geocentric
            latitudes are performed using the parameters of the IERS2003
            ellipsoid. Conversions between depth and radius assume the Earth is
            a sphere with radius 6371.
        dtype : str {'double', 'float', 'long', 'int', 'shortint', 'byte'}
            Data type used to store attribute values.
        description : str, optional
            Plain-language text description of the model.

        Attributes
        ----------
        grid : geotess.libgeotess.GeoTessGrid instance
        layers : tuple of Layer tuples
        attributes : tuple of Attribute tuples

        """
    def __init__(self, gridfile, layers=None, attributes=None, dtype=float,
            description=None):
        if dtype not in (float, int):
            raise ValueError("dtype must be float or int")

        self.layers = layers


    @classmethod
    def read(cls, modelfile):
        """
        Construct a Model instance from an existing model file name.

        """
        model = GeoTessModel.loadModel(modelfile)

    @property
    def layers(self):
        return self.metadata.layers

    def set_profile(self, vertex, layer, radii, attribute_values):
        """
        Set profile values for a vertex in a layer.

        """
        pass

    def write(self, outfile):
        """
        Write the instance to a file on disk.

        """
        pass

    def triangles(self, layer=None, level=None, connected=True):
        pass

    def vertices(self, layer=None, level=None, connected=True):
        pass

    def points(self, layer=None, level=None):
        pass

    def interpolate(self, domain, attribute, interpolant='linear'):
        pass

    def weights(self, domain, interpolant='linear'):
        pass

    def __str__(self):
        return self._model.toString()
