from pathlib import Path
from glob import glob

from setuptools import setup, Extension

import numpy as np

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False


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



if use_cython:
    # Every .pyx file in the the geotess directory will produce a lowecase Python extension module in the same location.
    pyxfiles = Path('geotess').glob('**/*.pyx')
    cy_extensions = [make_extension(pth) for pth in pyxfiles]

    compiler_directives = dict(
        embedsignature=True,
        language_level='3',
        c_string_type='unicode',
        c_string_encoding='utf-8',
    )
    extensions = cythonize(
        cy_extensions, 
        force=True, 
        compiler_directives=compiler_directives
    )
else:
    # Every .cpp file in the the geotess directory will produce a lowecase Python extension module in the same location.
    cppfiles = Path('geotess').glob('**/*.cpp')
    extensions = [make_extension(pth) for pth in cppfiles]


setup(name = 'pygeotess',
      version = '0.2.2',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      packages = ['geotess'],
      py_modules = ['geotess.model', 'geotess.exc'],
      ext_modules = extensions,
      # data_files = [
      #   ('geotess/data', glob('geotess/data/*.geotess')),
      #   ('geotess/data', glob('geotess/data/small_model_grid.ascii')),
      # ],
      package_data = {
        'geotess.data': [
            'geotess/data/*.geotess',
            'geotess/data/small_model_grid.ascii',
        ]
      },
      install_requires = [
          'numpy',
          'setuptools',
          ]
      )
