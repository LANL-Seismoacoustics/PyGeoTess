"""
GeoTessGrid Python definitions.

"""
import numpy as np

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
        if gridfile:
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

    def triangles(self, tess, level, masked=False):
        """
        Get tessellation triangles, as integer indices into the corresponding
        array of vertices.

        Use these "triangles" (vertex indices) to index into the corresponding
        Model.vertices.

        Parameters
        ----------
        tess : int
            The integer index of the target tessellation.
        level : int
            The integer of the target tessellation level.
        masked : bool
            If False, only return un-masked triangles.  Otherwise, return all.
            Not yet implemented.

        Returns
        -------
        triangles : numpy.ndarray of ints (Ntriangles x 3)
            Each row contains (unordered?) integer indices into the
            corresponding vertex array, producing the triangle coordinates.

        See Also
        --------
        Model.vertices

        """
        # XXX: handle None values from a grid.triangles() call.
        #   it should forward to an exposed _grid.getTriangles() method.

        # get the integer ids of all the triangles in this layer and level
        first_triangle_id = self._grid.getFirstTriangle(tess, level)
        last_triangle_id = self._grid.getLastTriangle(tess, level)
        triangle_ids = range(first_triangle_id, last_triangle_id + 1)

        # get the vertex indices of all the triangles as an iteger array
        triangles = np.empty((len(triangle_ids), 3), dtype=np.int)
        for i, triangle_id in enumerate(triangle_ids):
            triangles[i,:] = self._grid.getTriangleVertexIndexes(triangle_id)

        return triangles

    def vertices(self, tess=None, level=None, masked=False):
        """
        Get geocentric coordinates of tessellation vertices.

        Parameters
        ----------
        layer : str or int
            The string name or integer index of the target layer.
            Not yet implemented.
        level : int
            The integer of the target tessellation level.
            Not yet implemented.
        masked : bool
            If False, only return un-masked vertices.  Otherwise, return all.
            Not yet implemented.

        Returns
        -------
        vertices : numpy.ndarray of floats (Nvert X 2)
            Geographic coordinates of vertices.

        """
        # XXX: implement tess, level keywords.
        gt_util = lib.GeoTessUtils()
        unit_vertices = self._grid.getVertices()
        geo_vertices = np.empty((unit_vertices.shape[0], 2), dtype=np.float)
        for unit_vertex, geo_vertex in zip(unit_vertices, geo_vertices):
            # this syntax looks funny, because we never put geo_vertex back into
            # the original array, but it works b/c iterating through rows
            # in an array produces references.  so, manipulating these row
            # references is actually manipulating the main array.
            geo_vertex[0] = gt_util.getLonDegrees(unit_vertex)
            geo_vertex[1] = gt_util.getLatDegrees(unit_vertex)

        return geo_vertices


    def __str__(self):
        return str(self._grid.toString())
