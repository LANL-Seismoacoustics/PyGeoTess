try:
    import setuptools
except:
    pass

import os
import os.path
import sys
from glob import glob

waveforms = glob.glob('ndb/data/wf/E*')

setup(name = 'pygeotess',
      version = '0.1',
      description = 'GeoTess access for Python programmers.',
      author = 'Jonathan K. MacCarthy',
      author_email = 'jkmacc@lanl.gov',
      #install_requires=['pisces-db', 'numpy', 'matplotlib', 'basemap',
      #                  'obspy>=0.8', 'sqlalchemy>=0.7'],
      packages = ['geotess'],
      py_modules = ['geotess.utils', 'geotess.grid', 'geotess.model', 
                    'geotess.position', 'geotess.exc', 'geotess.data.models',
                    'geotess.data.grids'],
      data_files=[('geotess/lib', ['geotess/src/GeoTess.2.2.0.Java/geotess.jar'])],
      #entry_points = {
      #      'console_scripts': ['ndb_getstations': 'ndb.cli:ndb_getstations',
      #                          'ndb_getevents': 'ndb.cli:ndb_getevents',
      #                          'ndb_getwaveforms': 'ndb.cli:ndb_getwaveforms',
      #                          'ndb_plotstations': 'ndb.cli:ndb_plotstations',
      #                          'ndb_plotevents': 'ndb.cli:ndb_plotevents',
      #                          'ndb_plotwaveforms': 'ndb.cli:ndb_plotwaveforms',
      #                          ]
      #               }
      )

