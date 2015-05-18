"""

"""
from gov.sandia.geotess import GeoTessGrid

class Vector(object):
    @classmethod
    def from_coords(cls, lat, lon, depth):
        pass

class Triangle(object):
    pass


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
    def __init__(self, gridfile):
        """
        Construct a grid from a file.

        Parameters
        ----------
        gridfile : str
            Full path to GeoTess grid file.

        Attributes
        ----------
        grid : geotess.GeoTessGrid
            Low-level access to the GeoTessGrid instance.
        tesselations
        levels
        triangles
        vertices

        """
        grid = GeoTessGrid(gridfile)
        self.grid = grid

    def __eq__(self, value):
        return self.grid == value

    def __str__(self):
        return str(self.grid)

    def __repr__(self):
        return str(self.grid)

    # PROPERTIES
    # Properties are like getters.  Here, they're being used to create public
    # attributes from a subset of the underlying GeoTessGrid instance attributes.
    @property
    def tessellations(self):
        return self.grid.tessellations

    @property
    def levels(self):
        return self.grid.levels

    @property
    def triangles(self):
        return self.grid.triangles

    @property
    def vertices(self):
        return self.grid.vertices


    #@staticmethod
    #def from_grid(grid):
    #    """
    #    Return a copy of a Grid object.

    #    """
    #    return GeoTessGrid(grid.grid.tessellations, grid.grid.levels, 
    #                       grid.grid.triangles, grid.grid.vertices)

