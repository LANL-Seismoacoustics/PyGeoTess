#import os
#import sys
#
#sys.path.append(os.path.dirname(__file__) + os.path.sep + 'lib' + os.path.sep + 'geotess.jar')

__version__ = '0.3.0'

from geotess.utils import Layer, Attribute
from geotess.grid import Grid
from geotess.model import Model
