#
#
# sample file that includes the examples directory into the STIR build.
# You should copy this file to STIR/src/local (unless you have one there already, in 
# which case you need to edit that file instead of course).
# See also examples/README.txt and the section on how to extend STIR with your own 
# files in the STIR developer's Guide.


add_subdirectory(${PROJECT_SOURCE_DIR}/examples/src examples/src)