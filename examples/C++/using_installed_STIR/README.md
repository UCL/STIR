This directory contains some files that are intended for beginning STIR developers.
They illustrate how to use STIR in an external CMake project, as well as a minimal basic
C++ example.

-----------------

# Source code files

`CMakeLists.txt`: CMake file to use

`demo_create_image.cxx`: Simple program that creates an image with a shape.

# Build instructions
This works as normal for CMake. The only thing to take care of is to let
`find_package` find the installed STIR. Below uses the `cmake` command line
utility, but the GUI should work similarly, although there it is probably
most convenient to set `STIR_DIR` (see below).

## Configuring

If you installed STIR to `~/devel/install` and want to install this demo in the same place,
you should be able to do
```sh
cmake -S . -B build/  -DCMAKE_INSTALL_PREFIX:PATH=~/devel/install
```
If you want to install into a different location, you can do
```sh
cmake -S . -B build/   -DCMAKE_INSTALL_PREFIX:PATH=~/devel/install_demo  -DCMAKE_PREFIX_PATH:PATH=~/devel/install
```

If CMake doesn't find `STIRConfig.cmake` anyway, you will have to set `STIR_DIR=~/devel/install/lib/cmake/STIR-6.0` (or whatever version).

## Building and installing
Then you can
```sh
cmake --build build/ --config Release  --target install
```

# What now ?
Play around, incorporate demos in the other C++ examples and add them to the current `CMakeLists.txt`, ec.

Good luck

Kris Thielemans
