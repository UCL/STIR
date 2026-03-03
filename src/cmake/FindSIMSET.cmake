# FindSeimSET.cmake
# 
# Variables set: 
# SIMSET_FOUND - True if SimSET library found.
# SIMSET_INCLUDE_DIRS - where to find SimSET header files
# SIMSET_LIBRARY - the SimSET library to link against
#
# Targets: 
# SIMSET::simset

# Prefer SIMSET_ROOT_DIR from: 
# 1. CMake cache/command line: -DSIMSET_ROOT_DIR=...
# 2. Environment variable: SIMSET_ROOT_DIR
# 3. Environment variable: SIMSET

 if (NOT SIMSET_ROOT_DIR)
  if (NOT $ENV{SIMSET_ROOT_DIR} STREQUAL "")
    set(SIMSET_ROOT_DIR $ENV{SIMSET_ROOT_DIR})
  elseif (NOT $ENV{SIMSET} STREQUAL "")
    set(SIMSET_ROOT_DIR $ENV{SIMSET})
  endif()
endif()

if(SIMSET_ROOT_DIR)
  file(TO_CMAKE_PATH ${SIMSET_ROOT_DIR} SIMSET_ROOT_DIR)
endif()

find_path(SIMSET_INCLUDE_DIRS
      NAMES SystemDependent.h
      HINTS ${SIMSET_ROOT_DIR}
      PATH_SUFFIXES include Include src
      DOC "location of SIMSET include files")

find_library(SIMSET_LIBRARY
      NAMES simset libsimset
      HINTS ${SIMSET_ROOT_DIR}
      PATH_SUFFIXES lib Lib build lib64 obj
      DOC "Location of SIMSET library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  SIMSET
  REQUIRED_VARS SIMSET_LIBRARY SIMSET_INCLUDE_DIRS
  FAIL_MESSAGE "Could not find SIMSET include directories or library")  

if (SIMSET_FOUND AND NOT TARGET SIMSET::simset)
  add_library(SIMSET::simset UNKNOWN IMPORTED)
  set_target_properties(SIMSET::simset PROPERTIES
        IMPORTED_LOCATION ${SIMSET_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${SIMSET_INCLUDE_DIRS}
        )
  if (APPLE)
    target_compile_definitions(SIMSET::simset INTERFACE DARWIN MAC_10 GEN_UNIX arm64)
    elseif(UNIX)
      target_compile_definitions(SIMSET::simset INTERFACE GEN_UNIX)
      elseif(WIN32)
        target_compile_definitions(SIMSET::simset INTERFACE WIN32)
        endif()
endif()

if(SIMSET_LIBRARY)
  message(STATUS "SIMSET library found: ${SIMSET_LIBRARY}")
else()
  message(STATUS "SIMSET library not found.")
endif()