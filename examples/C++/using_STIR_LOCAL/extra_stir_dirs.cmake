#
#
# sample file that includes the examples directory into the STIR build.
# You should copy this file to STIR/src/local (unless you have one there already, in 
# which case you need to edit that file instead of course), or set the STIR_LOCAL CMake variable.
#
# See also examples/using_STIR_LOCAL/README.md and the section on how to extend STIR with your own
# files in the STIR developer's Guide.


add_subdirectory(${PROJECT_SOURCE_DIR}/examples/C++/using_STIR_LOCAL examples/C++/using_STIR_LOCAL)
