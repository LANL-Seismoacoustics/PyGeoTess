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

def _geotess_layer_names(name_list):
    """ Convert Python list of names to GeoTess-style name list.
    """
    # provide a list of string layer names
    # converts to uppercase, spaces to underscores, and joins them with "; "
    layer_names = '; '.join([name.upper().replace(' ', '_') for name in name_list])

    return layer_names

def _attributes_from_strings(gt_names, gt_units):
    """Produce geotess.Attribute tuples from GeoTess-style names and units lists.
    """
    names = gt_names.split(';')
    units = gt_units.split(';')

    return [Attribute(name=name, unit=unit) for name, unit in zip(names, units)]

def _geotess_attributes(attrib_tuples):
    """
    Convert geotess.Attribute tuples to GeoTess-style attribute name list and
    attribute unit list.

    """
    # Provide a list of geotess.util.Attribute tuples
    # No constraints on capitalization or spacing.
    attrib_names = "; ".join([attrib.name for attrib in attrib_tuples])
    attrib_units = "; ".join([attrib.unit for attrib in attrib_tuples])

    return attrib_names, attrib_units

def _layers_from_names_ids(gt_names, tess_ids):
    """Produce geotess.Layer tuples from GeoTess-style names and tess ids.
    """
    names = gt_names.split(';')

    return [Layer(name=name, tess_id=tess_id) for name, tess_id in zip(names, tess_ids)]

def _geotess_earth_shape(earth_shape, rconst):
    """Combine PyGeoTess earth shape and rconst values into GeoTess values.
    """
    if earth_shape == 'sphere' or not rconst:
        rc = ''
    else:
        rc = '_rconst'
    shp = earth_shape + rc

    return shp.upper()


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
        the extension should be ".ascii".  Because grid files are made using
        GeoTessBuilder and are required for a model, if gridfile is omitted,
        it is assumed that an empty Model class is desired, and the rest of the
        arguments are ignored.
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
    def __init__(self, gridfile=None, layers=None, attributes=None, dtype=None,
                 earth_shape="wgs84", rconst=False, description=None):

        if gridfile is None:
            pass
        else:
            self.layers = layers
            self.attributes = attributes

            # Create GeoTessMetaData from inputs
            md = lib.GeoTessMetaData()
            if not description:
                description = ""
            md.setDescription(description)

            layer_names = _geotess_layer_names([layer.name for layer in layers])
            md.setLayerNames(layer_names)
            md.setLayerTessIds([int(layer.tess_id) for layer in layers])

            attrib_names, attrib_units = _geotess_attributes(attributes)
            md.setAttributes(attrib_names, attrib_units)

            earth_shape = _geotess_earth_shape(earth_shape, rconst)
            md.setEarthShape(earth_shape)

            md.setDataType(dtype.upper())
            md.setModelSoftwareVersion("PyGeoTess v{}".format(__version__))
            md.setModelGenerationDate(str(datetime.now()))

            self._model = lib.GeoTessModel(gridfile, md)

    @classmethod
    def read(cls, modelfile):
        """
        Construct a Model instance from an existing model file.

        """
        m = cls()

        model = lib.GeoTessModel()
        model.loadModel(modelfile)
        m._model = model

        md = model.getMetaData()
        attribute_names = md.getAttributeNamesString()
        attribute_units = md.getAttributeUnitsString()
        attributes = _attributes_from_strings(attribute_names, attribute_units)
        m.attributes = attributes

        layer_names = md.getLayerNamesString()
        layer_tess_ids = md.getLayerTessIds()
        layers = _layers_from_names_ids(layer_names, layer_tess_ids)
        m.layers = layers

        return m


    def write(self, outfile):
        """
        Write the model to a file on disk.

        """
        self.model.writeModel(outfile)

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

