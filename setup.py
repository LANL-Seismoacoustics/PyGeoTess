from pathlib import Path
from glob import glob
import sys

from setuptools import setup, Extension

import numpy as np

from Cython.Build import cythonize


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
        library_dirs=[sys.prefix + '/lib'],
        libraries=['geotesscpp', 'geotessamplitudecpp'],
        include_dirs=[np.get_include()],
    )

    return extension



# Every .pyx file in the the geotess directory will produce a lowecase Python extension module in the same location.
pyxfiles = Path('geotess').glob('**/*.pyx')
cy_extensions = [make_extension(pth) for pth in pyxfiles]

compiler_directives = dict(
    embedsignature=True,
    language_level='3',
    c_string_type='unicode', # std::string outputs are coerced from bytes to Python 3 unicode str
    c_string_encoding='utf-8', # if std::string is coerced to Python 3 unicode str, use utf-8 decoding
)
extensions = cythonize(
    cy_extensions,
    force=True,
    compiler_directives=compiler_directives
)


setup(name = 'pygeotess',
      version = '0.2.2',
      description = 'GeoTess access from Python.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      zip_safe=False,
      packages = ['geotess', 'geotess.lib', 'geotess.data'],
      ext_modules = extensions,
      package_data = {
        'geotess.data': [
            '*.geotess',
            'small_model_grid.ascii',
        ]
      },
      install_requires = [
          'numpy',
          'setuptools',
          'cython',
          ]
      )
