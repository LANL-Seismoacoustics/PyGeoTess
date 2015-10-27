import platform
from glob import glob

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.sdist import sdist as _sdist

# see http://stackoverflow.com/a/4515279/745557
# and http://stackoverflow.com/a/18418524/745557

if platform.python_implementation() == 'CPython':
    CPPFILES = glob('geotess/src/GeoTessCPP/src/*.cc')
    PYXFILES = glob('geotess/*.pyx')
else:
    # Jython
    # in here will go code that deals with the GeoTess jar file
    CPPFILES = []
    PYXFILES = []

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    print("Cython detected.")
    class sdist(_sdist):
        def run(self):
            # Make sure the compiled Cython files in the distribution are up-to-date
            from Cython.Build import cythonize
            cythonize(['geotess/GeoTessGrid.pyx'])
            _sdist.run(self)

    ext_modules += [
        Extension(name='geotess.libgeotess', sources=CPPFILES+PYXFILES,
                  include_dirs=['geotess/src/GeoTessCPP/include']),
    ]
    cmdclass['build_ext'] = build_ext
    cmdclass['sdist'] = sdist
else:
    ext_modules += [
        Extension(name='geotess.libgeotess', sources=CPPFILES,
                  include_dirs=['geotess/src/GeoTessCPP/include']),
    ]


setup(name = 'pygeotess',
      version = '0.1',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      packages = ['geotess'],
      py_modules = ['geotess.grid', 'geotess.exc'],
      ext_modules = ext_modules,
      data_files = [ ('geotess/data', glob('geotess/src/GeoTessModels/*')) ],
      )

