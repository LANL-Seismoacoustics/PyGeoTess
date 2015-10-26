from glob import glob

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.sdist import sdist as _sdist

# see http://stackoverflow.com/a/4515279/745557
# and http://stackoverflow.com/a/18418524/745557

CPPFILES = glob('geotess/src/GeoTessCPP/src/*.cc')
HDRFILES = glob('geotess/src/GeoTessCPP/include/*.h')
PYXFILES = glob('geotess/*.pyx')

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    class sdist(_sdist):
        def run(self):
            # Make sure the compiled Cython files in the distribution are up-to-date
            from Cython.Build import cythonize
            cythonize(['geotess/GeoTessGrid.pyx'])
            _sdist.run(self)

    ext_modules += [
        Extension("geotess.libgeotesscpp", CPPFILES + HDRFILES + PYXFILES),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("geotess.libgeotesscpp", CPPFILES + HDRFILES),
    ]

cmdclass['sdist'] = sdist

setup(name = 'pygeotess',
      version = '0.1',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      #install_requires=['pisces-db', 'numpy', 'matplotlib', 'basemap',
      #                  'obspy>=0.8', 'sqlalchemy>=0.7'],
      packages = ['geotess'],
      py_modules = ['geotess.grid', 'geotess.exc'],
      )

