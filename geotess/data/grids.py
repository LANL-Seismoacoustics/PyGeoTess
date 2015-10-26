import os

from geotess.grid import Grid

here = os.path.dirname(__file__) + os.path.sep

grid_00500 = Grid(here + 'geotess_grid_00500.geotess')
grid_01000 = Grid(here + 'geotess_grid_01000.geotess')
grid_02000 = Grid(here + 'geotess_grid_02000.geotess')
grid_04000 = Grid(here + 'geotess_grid_04000.geotess')
grid_08000 = Grid(here + 'geotess_grid_08000.geotess')
grid_16000 = Grid(here + 'geotess_grid_16000.geotess')
grid_32000 = Grid(here + 'geotess_grid_32000.geotess')
grid_64000 = Grid(here + 'geotess_grid_64000.geotess')
