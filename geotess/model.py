"""
Home of the GeoTess Model.

Model is the main user-facing GeoTess class.  GeoTess Grids are build by 
geotessbuilder, and rarely (never) directly manipulated.  GeoTess MetaData
instances are mostly hidden within GeoTess Models.  The Model is the access
layer and manager of each of these other classes.

"""
from datetime import datetime

from geotess import __version__
from geotess.utils import Layer, Attribute
from geotess.grid import Grid
import geotess.libgeotess as lib

def _format_layer_names(name_list):
    # provide a list of string layer names
    # converts to uppercase and spaces to underscores
    layer_names = '; '.join([name.upper().replace(' ', '_') for name in name_list])

    return layer_names

def _format_attributes(attrib_tuples):
    # Provide a list of geotess.util.Attribute tuples
    # No constraints on capitalization or spacing.
    attrib_names = "; ".join([attrib.name for attrib in attrib_tuples])
    attrib_units = "; ".join([attrib.unit for attrib in attrib_tuples])

    return attrib_names, attrib_units


class Model(object):
    """
    A class representing a 2D or 3D gridded Earth model.

    A GeoTess Model is comprised of 2D triangular tessellations of a unit
    sphere with 1D radial arrays of nodes associated with each vertex of
    the 2D tessellations. Variable spatial resolution in both geographic
    and radial dimensions is supported. 

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
    dtype : str {'double', 'float', 'long', 'int', 'shortint', 'byte'}
        Data type used to store attribute values.
    earth_shape : str {"sphere", "grs80", "wgs84"}
        Defines the ellipsoid that is to be used to convert between
        geocentric and geographic latitude and between depth and radius.
        The default is wgs84.

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
    rconst : bool
        If True, assume, for purposes of converting between depth and
        radius, that the radius of the earth is a constant equal
        to 6371 km.  Otherwise, assume that the radius decreases from
        equator to poles according to the specified earth_shape. 
    description : str, optional
        Plain-language description of the model.

    Attributes
    ----------
    layers : tuple of Layer tuples
    attributes : tuple of Attribute tuples

    References
    ----------
    http://www.sandia.gov/geotess

    """
    def __init__(self, gridfile, layers, attributes, dtype, earth_shape="wgs84",
                 rconst=True, description=None):

        self.layers = layers
        self.attributes = attributes

        grid = lib.GeoTessGrid()
        grid.loadGrid(gridfile)
        self.grid = grid

        # Create GeoTessMetaData from inputs
        md = lib.GeoTessMetaData()
        if not description:
            description = ""
        md.setDescription(description)

        layer_names = _format_layer_names([layer.name for layer in layers])
        tess_ids = [int(layer.tess_id) for layer in layers]
        md.setLayerNames(layer_names)
        md.setLayerTessIds(tess_ids)

        attrib_names, attrib_units = _format_attributes(attributes)
        md.setAttributes(attrib_names, attrib_units)

        md.setDataType(dtype.upper())
        md.setModelSoftwareVersion("PyGeoTess v{}".format(__version__))
        md.setModelGenerationDate(str(datetime.now()))
        self.metadata = md

        self.model = lib.GeoTessModel(grid, md)

    @classmethod
    def read(cls, modelfile):
        """
        Construct a Model instance from an existing model file.

        """
        model = lib.GeoTessModel.loadModel(modelfile)

    def write(self, outfile):
        """
        Write the instance to a file on disk.

        """
        pass

    def triangles(self, layer=None, level=None, connected=True):
        """
        Get tessellation triangles, as integer indices into the corresponding
        array of vertices.

        Parameters
        ----------
        layer, level : str or int
            The string name or integer index of the target layer, level.
        connected : bool
            If True, only return connected triangles.

        Returns
        -------
        triangles : numpy.ndarray of ints (Ntriangles x 3)
            Each row contains integer indices into the corresponding vertex
            array, producing the triangle coordinates.

        See Also
        --------
        Model.vertices

        """
        pass

    def vertices(self, layer=None, level=None, connected=True):
        """
        Get geographic coordinates of tessellation vertices.

        Parameters
        ----------
        layer, level : str or int
            The string name or integer index of the target layer, level.
        connected : bool
            If True, only return connected triangles.

        Returns
        -------
        vertices : numpy.ndarray of floats (Nvert X Ndim)
            Geographic coordinates of vertices.
            For 2D models, vertices is Nvert X 2 (lon, lat).
            For 3D models, vertices is Nvert X 3 (lon, lat, radius).

        """
        pass

    def points(self, layer=None, level=None):
        """
        Get geographic coordinates of model points.

        Parameters
        ----------
        layer, level : str or int
            The string name or integer index of the target layer, level.

        Returns
        -------
        points : numpy.ndarray of floats (Nvert X Ndim)
            Geographic coordinates of points.
            For 2D models, vertices is Nvert X 2 (lon, lat).
            For 3D models, vertices is Nvert X 3 (lon, lat, radius).

        """
        pass

    def interpolate(self, domain, attribute, interpolant='linear'):
        pass

    def weights(self, domain, interpolant='linear'):
        pass

    def set_profile(self, vertex, layer, radii, attribute_values):
        """
        Set profile values for a vertex in a layer.

        """
        pass

    def __str__(self):
        return self.model.toString()

