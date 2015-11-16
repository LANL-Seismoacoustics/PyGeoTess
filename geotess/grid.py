"""
GeoTessGrid Python definitions.

"""
import geotess.libgeotess as lib

class Grid(object):
    """
    Manages the geometry and topology of a multi-level triangular tessellation
    of a unit sphere. It knows:

    * the positions of all the vertices,
    * the connectivity information that defines how vertices are connected to
      form triangles,
    * for each triangle it knows the indexes of the 3 neighboring triangles,
    * for each triangle it knows the index of the triangle which is a descendant
      at the next higher tessellation level, if there is one.
    * information about which triangles reside on which tessellation level 

    """
    def __init__(self, gridfile=None):
        """
        Construct a grid from a file.

        Parameters
        ----------
        gridfile : str
            Full path to GeoTess grid file.  None results in an empty Grid
            instance.

        Attributes
        ----------
        _grid : geotess.GeoTessGrid
            Low-level access to the GeoTessGrid instance.
        tesselations
        levels
        triangles
        vertices

        """
        if gridfile
            self._grid = lib.GeoTessGrid()
            self._grid.loadGrid(gridfile)
        else:
            self._grid = None

    @classmethod
    def from_geotessgrid(cls, gtgrid):
        """
        Constructor that wraps a geotess.libgeotess.GeoTessGrid instance.

        """
        g = cls()
        g._grid = gtgrid

        return g

    def triangles(self, tessellation=None, level=None, masked=None):
        # TODO: copy Model.triangles over here.  Have Model.triangles forward to 
        #    Model.grid.triangles.
        pass

    def __str__(self):
        return str(self._grid.toString())
