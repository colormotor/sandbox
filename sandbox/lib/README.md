# Autograff utilities bindings
Python bindings for the [Colormotor](https://github.com/colormotor/colormotor) framework,
built with [SWIG](http://www.swig.org) and a modified version of [ArmanPy](https://sourceforge.net/projects/armanpy/) for interoperability with NumPy. 

## Prerequisites

First generate the Colormotor Python bindings by following the instructions here:
[https://github.com/colormotor/colormotor/tree/master/addons/pycolormotor](https://github.com/colormotor/colormotor/tree/master/addons/pycolormotor)

## Compiling
First generate the bindings with

```
sh swig.sh
```

Then compile with:

```
mkdir build; cd build
cmake ..
make
```

This will generate the python wrappers and dynamic libraries that will be located in the parent directory.

## Notes
### Armadillo/NumPy
The wrapper utilises a modified version of [ArmanPy](https://sourceforge.net/projects/armanpy/) to allow transparent conversion between Armadillo and NumPy arrays/matrices. The provided version of ArmanPy automatically converts NumPy arrays to the same memory layout as Armadillo (columns linear in memory, in NumPy terms FORTRAN-contiguous format), which may result in slower performance due to the memory copy, but simplifies integration and binding of Armadillo code in Python.
