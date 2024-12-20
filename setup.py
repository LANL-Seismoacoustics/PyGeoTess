from glob import glob

from setuptools import setup, Extension

import numpy as np

PYXFILES = ['geotess/src/libgeotess.pyx'] # hand-crafted Cython (automatically finds clibgeotess.pxd)
CYFILES = ['geotess/src/libgeotess.cpp'] # pre-cythonized c++ source files, in case cythonize fails

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

if use_cython:
    extensions = [Extension(name='geotess.libgeotess',
                  sources=PYXFILES, language='c++',
                  libraries=['geotesscpp', 'geotessamplitudecpp'],
                  # library_dirs=LIBDIRS,
                  # extra_link_args=LDFLAGS,
                  include_dirs=[np.get_include()])]
    extensions = cythonize(extensions, force=True)
else:
    extensions = [Extension(name='geotess.libgeotess',
                  sources=CYFILES, language='c++',
                  libraries=['geotesscpp', 'geotessamplitudecpp'],
                  # library_dirs=LIBDIRS,
                  # extra_link_args=LDFLAGS,
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
          'setuptools',
          ]
      )
