# PyGeoTess

GeoTess for Python programmers.

PyGeoTess is a Python interface module to the
[GeoTess](http://www.sandia.gov/geotess) gridding and earth model library from
Sandia National Laboratories.  It provides two interfaces to a subset of the
GeoTess library: a direct interface to the GeoTess c++ classes and methods,
and a more Pythonic interface.

![global grid](docs/src/pages/data/output_9_1.png)


## Installation

PyGeoTess currently requires a C++ compiler.  In the future, binary wheels may be available on PyPI.

To install centrally from this repo: `pip install .`

To install an "editable" local installation from this repo: `pip install -e .`
