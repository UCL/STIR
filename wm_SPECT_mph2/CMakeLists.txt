cmake_minimum_required(VERSION 2.7)

PROJECT(wm_spect_mph)

#Add the *.cpp of the *.h files
add_library(wm_SPECT  STATIC src/weight3d_SPECT_mph.cpp src/wmtools_SPECT_mph.cpp )

#Include the directory of *.h files
include_directories ("include")

ADD_EXECUTABLE(wm_SPECT_mph wm_SPECT_mph.cpp)

#Link the main source with the libraries (*.h)
target_link_libraries(wm_SPECT_mph wm_SPECT)