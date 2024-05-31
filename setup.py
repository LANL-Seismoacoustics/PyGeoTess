from glob import glob
import sys

from distutils.core import setup
from distutils.extension import Extension

import numpy as np

# see http://stackoverflow.com/a/4515279/745557
# and http://stackoverflow.com/a/18418524/745557

CPPFILES = glob('geotess/src/*.cc') # GeoTess c++ source (automatically finds .h files)
PYXFILES = ['geotess/src/libgeotess.pyx'] # hand-crafted Cython (automatically finds clibgeotess.pxd)
CYFILES = glob('geotess/src/libgeotess.cpp') # pre-cythonized c++ source files, in case cythonize fails

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

if use_cython:
    extensions = [Extension(name='geotess.libgeotess',
                  sources=CPPFILES+PYXFILES, language='c++',
                  include_dirs=[np.get_include()])]
    extensions = cythonize(extensions)
else:
    extensions = [Extension(name='geotess.libgeotess',
                  sources=CPPFILES+CYFILES, language='c++',
                  include_dirs=[np.get_include(), 'geotess/src'])]


setup(name = 'pygeotess',
      version = '0.2.2',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      packages = ['geotess'],
      py_modules = ['geotess.model', 'geotess.exc'],
      ext_modules = extensions,
      data_files = [ ('geotess/data', glob('geotess/src/GeoTessModels/*')) ],
      install_requires = [
          'numpy',
          ]
      )

