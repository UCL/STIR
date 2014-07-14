# BuildMex.cmake
# Author: Kent Williams norman-k-williams@...
# Modified by Kris Thielemans for WIN32
include(CMakeParseArguments)

if(NOT MATLAB_FOUND)
  find_package(MATLAB REQUIRED)
endif()

if(NOT MATLAB_MEX_PATH)
  message(FATAL_ERROR "Can't find Matlab MEX compiler")
endif()

include_directories(${MATLAB_INCLUDE_DIR})

#
# BuildMex -- arguments
# MEXNAME = root of mex library name
# TARGETDIR = location for the mex library files to be created
# SOURCE = list of source files
# LIBRARIES = libraries needed to link mex library
macro(BuildMex)
  set(oneValueArgs MEXNAME TARGETDIR)
  set(multiValueArgs SOURCE LIBRARIES)
  cmake_parse_arguments(BuildMex "" "${oneValueArgs}" "${multiValueArgs}"
${ARGN})
  set_source_files_properties(${BuildMex_SOURCE}    COMPILE_DEFINITIONS ${mexCxxFlags}    )
  add_library(${BuildMex_MEXNAME} SHARED ${BuildMex_SOURCE})
  set_target_properties(${BuildMex_MEXNAME} PROPERTIES
    SUFFIX "${MATLAB_MEX_EXT}"
    LINK_FLAGS "${mexLdFlags}"
    RUNTIME_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    )
  target_link_libraries(${BuildMex_MEXNAME} ${BuildMex_LIBRARIES} ${mexCxxLibs})
endmacro(BuildMex)
