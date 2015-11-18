import os

from geotess.model import Model

here = os.path.dirname(__file__) + os.path.sep

crust10 = Model.read(here + 'crust10.geotess')
crust20 = Model.read(here + 'crust20.geotess')
