from pathlib import Path
from glob import glob

from setuptools import setup, Extension

import numpy as np

# PYXFILES = ['geotess/src/libgeotess.pyx'] # hand-crafted Cython (automatically finds clibgeotess.pxd)
# CYFILES = ['geotess/src/libgeotess.cpp'] # pre-cythonized c++ source files, in case cythonize fails

# extensions = [ext for ext in make_extension(pathlib.Path('.').glob('geotess/*.pyx'))]
def make_extension(extpath: Path) -> Extension:
    """ Make a setuptools.Extension from a Path to a pygeotess source file. 

    The extension location/name is a mirros the original pyx file location, with a lowercase stem.

    Examples
    --------
    >>> make_extension(pathlib.Path('geotess/lib/GeoTessModel.pyx'))
    <setuptools.extension.Extension('geotess.lib.geotessutils') at 0x107402900>

    >>> make_extension(pathlib.Path('GeoTess/Lib/GeoTessModel.cpp'))
    <setuptools.extension.Extension('GeoTess.Lib.geotessutils') at  0x10aba2fc0>

    """
    extension = Extension(
        name='.'.join(extpath.parts[:-1] + (extpath.stem.lower(),)),
        sources=[str(extpath)],
        language='c++',
        libraries=['geotesscpp', 'geotessamplitudecpp'],
        include_dirs=[np.get_include()],
    )

    return extension

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

compiler_directives = dict(
    embedsignature=True,
    language_level='3',
    c_string_type='unicode',
    c_string_encoding='utf-8',
)

if use_cython:
    extensions = [make_extension(pth) for pth in Path('.').glob('geotess/**/*.pyx')]
    # a single lib extension module containing all Cython implementation classes
    # extensions = [
    #     Extension(
    #         name='geotess.lib',
    #         # sources=["geotess/lib/*.pyx"],
    #         sources=glob("geotess/*.pyx"),
    #         language='c++',
    #         libraries=['geotesscpp', 'geotessamplitudecpp'],
    #         include_dirs=[np.get_include()],
    #     )
    # ]
    extensions = cythonize(
        extensions, 
        force=True, 
        compiler_directives=compiler_directives
    )
else:
    extensions = [make_extension(pth) for pth in Path('.').glob('geotess/**/*.cpp')]
    # extensions = [
    #     Extension(
    #         name='geotess.lib',
    #         # sources=["geotess/lib/*.cpp"],
    #         sources=glob("geotess/*.cpp"),
    #         language='c++',
    #         libraries=['geotesscpp', 'geotessamplitudecpp'],
    #         include_dirs=[np.get_include(), 'geotess'],
    #     )
    # ]


setup(name = 'pygeotess',
      version = '0.2.2',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      packages = ['geotess'],
      py_modules = ['geotess.model', 'geotess.exc'],
      ext_modules = extensions,
      data_files = [ ('geotess/data', glob('GeoTessModels/*')) ],
      install_requires = [
          'numpy',
          'setuptools',
          ]
      )
