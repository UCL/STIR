# BuildMex.cmake
# Author: Kent Williams norman-k-williams@...
# Modified by Kris Thielemans for WIN32
include(CMakeParseArguments)

if(NOT MATLAB_FOUND)
  find_package(MATLAB REQUIRED)
endif()
# CMake 2.8.12 & earlier apparently don't define the
# Mex script path, so find it.
if(NOT MATLAB_MEX_PATH)
  find_program( MATLAB_MEX_PATH mex
    HINTS ${MATLAB_ROOT}/bin
    PATHS ${MATLAB_ROOT}/bin
    DOC "The mex program path"
    )
  if(NOT MATLAB_MEX_PATH)
    message(FATAL_ERROR "Can't find Matlab MEX compiler")
  endif()
endif()

include_directories(${MATLAB_INCLUDE_DIR})
#
# mex -v outputs all the settings used for building MEX files, so it
# we can use it to grab the important variables needed to generate
# a well formed mex file.
execute_process(COMMAND ${MATLAB_MEX_PATH} -v
  OUTPUT_VARIABLE mexOut
  ERROR_VARIABLE mexErr)

# parse mex output line by line by turning file into CMake list of lines
string(REGEX REPLACE "\r?\n" ";" _mexOut "${mexOut}")
foreach(line ${_mexOut})  
if (WIN32)
  if("${line}" MATCHES " COMPFLAGS *=")
    string(REGEX REPLACE " *COMPFLAGS *= *" "" mexCxxFlags "${line}")
  elseif("${line}" MATCHES " *LINKFLAGS *=")
    string(REGEX REPLACE " *LINKFLAGS *= *" "" mexLdFlags "${line}")
    # get rid of /implib statement (refers to temp file)
    string(REGEX REPLACE "/implib:\".*\"" "" mexLdFlags "${mexLdFlags}")
  elseif("${line}" MATCHES " *Name directive *=")
    string(REGEX REPLACE " *Name directive *= */out:\"(.*)\"" "\\1" mexLdExtension "${line}")
  endif()
else()
  if("${line}" MATCHES " CXXFLAGS *=")
    string(REGEX REPLACE " *CXXFLAGS *= *" "" mexCxxFlags "${line}")
  elseif("${line}" MATCHES " CXXLIBS *=")
    string(REGEX REPLACE " *CXXLIBS *= *" "" mexCxxLibs "${line}")
  elseif("${line}" MATCHES " LDFLAGS *=")
    string(REGEX REPLACE " *LDFLAGS *= *" "" mexLdFlags "${line}")
  elseif("${line}" MATCHES " LDEXTENSION *=")
    string(REGEX REPLACE " *LDEXTENSION *= *" "" mexLdExtension "${line}")
  endif()
endif()
endforeach()

#list(APPEND mexCxxFlags "-DMATLAB_MEX_FILE")

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
    SUFFIX "${mexLdExtension}"
    LINK_FLAGS ${mexLdFlags}
    RUNTIME_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    )
  target_link_libraries(${BuildMex_MEXNAME} ${BuildMex_LIBRARIES} ${mexCxxLibs})
endmacro(BuildMex)
