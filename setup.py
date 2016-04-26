import platform
from glob import glob

from distutils.core import setup
from distutils.extension import Extension

import numpy as np

# see http://stackoverflow.com/a/4515279/745557
# and http://stackoverflow.com/a/18418524/745557

if platform.python_implementation() == 'CPython':
    CPPFILES = glob('geotess/src/*.cc') # GeoTess c++ source
    # PYXFILES = glob('geotess/src/*.pyx') # Cython source files
    # CYFILES = glob('geotess/*.cpp') # cythonized c++ source files
    PYXFILES = ['geotess/src/libgeotess.pyx']
    CYFILES = glob('geotess/src/libgeotess.cpp') # cythonized c++ source files
else:
    # Jython
    # in here will go code that deals with the GeoTess jar file
    CPPFILES = []
    PYXFILES = []
    CYFILES = []

try:
    from Cython.Built import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True


if use_cython:
    extensions = [Extension(name='geotess.libgeotess',
                  sources=CPPFILES+PYXFILES, language='c++',
                  include_dirs=[np.get_include()])]
    extensions = cythonize(extensions, language='c++')
else:
    extensions = [Extension(name='geotess.libgeotess',
                  sources=CPPFILES+CYFILES, language='c++',
                  include_dirs=[np.get_include(), 'geotess/src'])]


setup(name = 'pygeotess',
      version = '0.2.1',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      packages = ['geotess'],
      py_modules = ['geotess.model', 'geotess.exc'],
      ext_modules = extensions,
      data_files = [ ('geotess/data', glob('geotess/src/GeoTessModels/*')) ],
      )

