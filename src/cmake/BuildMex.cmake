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
# mex -v outputs all the settings used for building MEX files, so it
# we can use it to grab the important variables needed to generate
# a well formed mex file.
execute_process(COMMAND ${MATLAB_MEX_PATH} -v -n ${PROJECT_SOURCE_DIR}/src/cmake/FindMATLAB_mextest.c
  OUTPUT_VARIABLE mexOut
  ERROR_VARIABLE mexErr)

# parse mex output line by line by turning file into CMake list of lines
string(REGEX REPLACE "\r?\n" ";" _mexOut "${mexOut}")
foreach(line ${_mexOut})  
if (WIN32)
  if("${line}" MATCHES "[\t ]*COMPDEFINES *:")
    string(REGEX REPLACE "[\t ]*COMPDEFINES *: *" "" mexCxxFlags "${line}")
  elseif("${line}" MATCHES "[\t ]*INCLUDE *:")
    string(REGEX REPLACE "[\t ]*INCLUDE *: *" "" mexIncludeFlags "${line}")
  elseif("${line}" MATCHES "[\t ]*LINKFLAGS *:")
    string(REGEX REPLACE "[\t ]*LINKFLAGS *: *" "" mexLdFlags "${line}")
  elseif("${line}" MATCHES "[\t ]*LINKLIBS *:")
    string(REGEX REPLACE "[\t ]*LINKLIBS *: *" "" mexLdLibs "${line}")
  elseif("${line}" MATCHES "[\t ]*LINKEXPORT *:")
    string(REGEX REPLACE "[\t ]*LINKEXPORT *: *" "" mexLdExport "${line}")
    # get rid of /implib statement (refers to temp file)
    string(REGEX REPLACE "/implib:\".*\"" "" mexLdFlags "${mexLdFlags}")
  endif()
else()
  if("${line}" MATCHES " CXXFLAGS *=")
    string(REGEX REPLACE " *CXXFLAGS *= *" "" mexCxxFlags "${line}")
  elseif("${line}" MATCHES " CXXLIBS *=")
    string(REGEX REPLACE " *CXXLIBS *= *" "" mexCxxLibs "${line}")
  elseif("${line}" MATCHES " LDFLAGS *=")
    string(REGEX REPLACE " *LDFLAGS *= *" "" mexLdFlags "${line}")
  endif()
endif()
endforeach()

if (WIN32)
  # note: cannot use include flags for "include_directories" as that gets confused
  # by the -I notation. So, instead we add it to the compiler flags
  set(mexCxxFlags "${mexCxxFlags} ${mexIncludeFlags}")
  # note: cannot use mexLdLibs for "target_link_libraries" as that gets confused 
  # by the flag that specifies where the matlab libraries are. So, instead we add
  # it to the linker flags
  set(mexLdFlags "${mexLdFlags} ${mexLdLibs} ${mexLdExport}")
endif()

message(STATUS "mexLdFlags: ${mexLdFlags}")
message(STATUS "mexCxxFlags: ${mexCxxFlags}")
message(STATUS "mexCxxLibs: ${mexCxxLibs}")
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
    SUFFIX "${MATLAB_MEX_EXT}"
    LINK_FLAGS "${mexLdFlags}"
    RUNTIME_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    )
  target_link_libraries(${BuildMex_MEXNAME} ${BuildMex_LIBRARIES} ${mexCxxLibs})
endmacro(BuildMex)
