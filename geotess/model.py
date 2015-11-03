"""
Home of GeoTessModel.

GeoTessModel is the main user-facing class.  GeoTessGrids are build by 
geotessbuilder, and rarely (never) directly manipulated.  (GeoTess)MetaData
instances are mostly hidden within GeoTessModels.  The GeoTessModel is the access
layer and manager of each of these other classes.

"""
from collections import namedtuple

from geotess.grid import Grid

from geotess.libgeotess import GeoTessModel

# These namedtuple are lightweight readable containers for Layer/Attribute info
Layer = namedtuple('Layer', ['name', 'tess_id'])
Attribute = namedtuple('Attribute', ['name', 'unit'])

class Model(object):
    def __init__(self, gridfile, layers=None, attributes=None, dtype=None, description=None):
        """
        Initialize a GeoTessModel using a gridfile and optional metadata information.

        Parameters
        ----------
        gridfile : str
            Full path to an existing GeoTess grid file.
        layers : list or tuple of Layers
            A sequence Layer named tuples describing model layers and tess_ids.
        attributes : list or tuple of Attributes
            A sequence of Attribute name tuples describing model attributes and units.
        dtype : {float, int}
            Data type used to store attribute values.
        description : str, optional
            Plain english description of the model.

        Attributes
        ----------
        grid : geotess.GeoTessGrid instance
        metadata : geotess.MetaData instance
        layers : list of Layers
        description : str

        """
        grid = Grid(gridfile)
        metadata = MetaData(layers, attributes, dtype, description)

        self.grid = grid
        self.metadata


    @classmethod
    def read(cls, modelfile):
        """
        Construct a GeoTessModel instance from an existing model file.

        """
        # XXX
        model = GeoTessModel(modelfile)

    @property
    def layers(self):
        return self.metadata.layers

    @property
    def description(self):
        return self.metadata.description

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
