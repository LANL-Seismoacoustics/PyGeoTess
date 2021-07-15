# Contributing to PyGeoTess

Thanks for your interest in contributing to PyGeoTess.
The main areas that need help are:

* wrapping new [GeoTessCPP](https://github.com/sandialabs/GeoTessCPP) classes and methods.
* designing and implementing a more Pythonic way to use PyGeoTess
* improving the documentation

## To wrap new GeoTessCPP functionality

1. If you're not in the Seismoacoustics group, fork this repo.
2. Make a new branch for your additions.
3. Declare your class and/or method in `geotess/src/clibgeotess.pxd`, following the examples there.  The way I do this is to look at the [GeoTessCPP API docs](https://www.sandia.gov/RSTT/software/documentation/geotess/docs/index.html) (which is currently more up-to-date in the RSTT project than the GeoTessCPP project), and find the class / method signature and (generally) just copy it into `clibgeotess.pxd`.
4. Next, implement the class / method in `geotess/src/libgeotess.pyx` using Cython, again, following the examples there.  For simple classes / methods, this could be as simple as forwarding the method arguments directly to the method on the class pointer, like with `GeoTessGrid.getVertexIndex`.
5. Make a test for your class and/or method in the `tests` directory.
6. Push your changes to GitHub.
7. Make a Pull Request to the main PyGeoTess branch, and watch to make sure the tests pass.
8. If the tests fail, look at the test logs on GitHub to help diagnose the problem. To develop locally, you need to `pip install -U -e .` when you make changes to Cython files (so that they're re-compiled and re-installed), make sure Pytest is installed, and run `pytest tests` from the main project directory.
