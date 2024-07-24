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

### GeoTessCPP

First, install GeoTessCPP >= 2.7, the underlying C++ library powering PyGeoTess,
available from Conda-Forge or directly from the [SNL repository](https://github.com/sandialabs/GeoTessCPP):

With conda:

```bash
conda install -c conda-forge geotesscpp`
```

NOTE: Using PyGeoTess with `geotesscpp` installed from the main SNL repository does not currently work.


### PyGeoTess

To install centrally from this repo: `pip install .`

To install an "editable" local installation from this repo: `pip install -e .`
