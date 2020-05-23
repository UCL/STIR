# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
cmake_minimum_required(VERSION 2.8.3)
# require 2.8.3 to get FOLDER properties support (without that, we only need cmake 2.6)

# avoid warning about WIN32 no longer defined in CYGWIN
set(CMAKE_LEGACY_CYGWIN_WIN32 0) 

# enable ccache https://ccache.samba.org/
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
  message(STATUS "ccache found, so we will use this.")
endif()

PROJECT(STIR)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# require 2.8.3 to get FOLDER properties support (without that, we only need cmake 2.6)
cmake_minimum_required(VERSION 2.8.3)
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW) # Compiler Id for Apple Clang is AppleClang, see STIR issue #531
endif()
  
# add project source to cmake path such that it can use our find_package modules and .cmake files
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/cmake;${CMAKE_MODULE_PATH}")
include(src/cmake/SetC++Version.cmake)

UseCXX(11)

# set default build-type to Release
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "type of build: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if((${CMAKE_CXX_COMPILER_ID} MATCHES "AppleClang") OR
   (APPLE AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND
    ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  # Ignore undefined template var warning
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")
endif()

####### Set Version number etc
set(VERSION_MAJOR   4)
set(VERSION_MINOR   0)
set(VERSION_PATCH   2)
set(VERSION   040002) # only used in STIRConfig.h.in

set(STIR_VERSION
  ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

####### Installation directories
# Use locations as in
# https://gitlab.kitware.com/cmake/cmake/blob/master/Source/CMakeInstallDestinations.cmake
# TODO append version numbers
if(BEOS)
  set(STIR_DOC_DIR "documentation/doc/stir")
elseif(CYGWIN)
  set(STIR_DOC_DIR "share/doc/stir")
else()
  set(STIR_DOC_DIR "doc/stir")
endif()
set(STIR_DATA_DIR "share/stir")
set(ConfigPackageLocation lib/cmake/)

####### External packages
# Note: we need to have the find_package statements in the top-level CMakeLists.txt
# such that we can use it in STIRConfig.cmake.in (see below).

#### we need the boost library from boost.org
set(BOOST_ROOT CACHE PATH "root of Boost")
find_package( Boost 1.36.0 REQUIRED )

#### optional external libraries. 
# Listed here such that we know if we should compile extra utilities
option(DISABLE_LLN_MATRIX "disable use of LLN library" OFF)
option(DISABLE_ITK "disable use of ITK library" OFF)
option(DISABLE_AVW "disable use of AVW library" OFF)
option(DISABLE_RDF "disable use of GE RDF library" OFF)
option(DISABLE_STIR_LOCAL "disable use of LOCAL extensions to STIR" OFF)
option(DISABLE_CERN_ROOT "disable use of Cern ROOT libraries" OFF)
option(DISABLE_NLOHMANN_JSON "disable use of nlohmann JSON libraries" OFF)
option(STIR_ENABLE_EXPERIMENTAL "disable use of STIR experimental code" OFF) # disable by default
option(DISABLE_NiftyPET_PROJECTOR "disable use of NiftyPET projector" OFF)

if(NOT DISABLE_ITK)
   # See if we can find a compiled version of ITK (http://www.itk.org/)
   find_package(ITK QUIET)
   if (ITK_FOUND)
      include(${ITK_USE_FILE})
   endif()
endif()

if(NOT DISABLE_LLN_MATRIX)
  find_package(LLN)
endif()

if(NOT DISABLE_CERN_ROOT)
  find_package(CERN_ROOT)
  if (CERN_ROOT_FOUND)
    message(STATUS "ROOT Version: ${CERN_ROOT_VERSION}")
    # Find which major version this is
    string(REPLACE "." ";" _VERSION_LIST ${CERN_ROOT_VERSION})
    list(GET _VERSION_LIST 0 CERN_ROOT_VERSION_MAJOR)
    if (${CERN_ROOT_VERSION_MAJOR} GREATER 5)
      message(STATUS "This ROOT version needs C++-11, so we will enable this.")
      UseCXX(11)
    endif()
  endif()
endif()

if(NOT DISABLE_AVW)
  find_package(AVW)
endif()

if(NOT DISABLE_RDF)
  find_package(RDF)
endif()

if(NOT DISABLE_NLOHMANN_JSON)
    find_package(nlohmann_json 3.2.0 QUIET)
    if (nlohmann_json_FOUND)
        message(STATUS "nlohmann JSON library found.")
        message(STATUS "The nlohmann JSON library needs C++-11, so we will enable this.")
        UseCXX(11)
    endif()
endif()

# Enable version 3 projectors
option(STIR_PROJECTORS_AS_V3 "Enable STIR version 3 legacy code" OFF)

# NiftyPET projector
if(NOT DISABLE_NiftyPET_PROJECTOR)
  find_package(CUDAToolkit QUIET)
  if (CUDAToolkit_FOUND)
    find_package(NiftyPET)
    if (NiftyPET_FOUND)
      set(STIR_WITH_NiftyPET_PROJECTOR ON)
    endif()
  endif()
endif()
if (STIR_WITH_NiftyPET_PROJECTOR)
  message(STATUS "NiftyPET projector support enabled.")
else()
  message(STATUS "NiftyPET projector support disabled.")
endif()

#### enable support for ctest
ENABLE_TESTING()

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(scripts)

#### export configuration for external projects that want to use STIR
# See https://cmake.org/cmake/help/v3.0/manual/cmake-packages.7.html
# Also https://rix0r.nl/blog/2015/08/13/cmake-guide/
# https://coderwall.com/p/qej45g/use-cmake-enabled-libraries-in-your-cmake-project-iii

include(CMakePackageConfigHelpers)

WRITE_BASIC_PACKAGE_VERSION_FILE(${CMAKE_CURRENT_BINARY_DIR}/STIRConfigVersion.cmake
                                 VERSION ${STIR_VERSION}
                                 COMPATIBILITY SameMajorVersion )

# create STIRTargets*.cmake files for importing the build-tree (disabled for now)
#export(EXPORT STIRTargets
#  FILE "${CMAKE_BINARY_DIR}/STIRTargets.cmake"
#)

## create files specific to the "installed" version

# Set STIR_INCLUDE_DIRS before exporting such that it will refer to
# the installed files, not the source.
set (STIR_INCLUDE_DIRS "include")

# install the registry sources
install(FILES ${STIR_REGISTRIES} DESTINATION ${STIR_DATA_DIR}/src)

# set STIR_REGISTRIES to this location before export
# first create a new variable.
set(INSTALLED_STIR_REGISTRIES)
foreach(r  ${STIR_REGISTRIES})
  # get filename without directory
  get_filename_component(r_filename ${r} NAME)
  # append this with installed location
  # (note: have to use CMAKE_INSTALL_PREFIX for CONFIGURE_PACKAGE_CONFIG_FILE to
  # make the variable relocatable)
  list(APPEND INSTALLED_STIR_REGISTRIES "${CMAKE_INSTALL_PREFIX}/${STIR_DATA_DIR}/src/${r_filename}")
endforeach()
set(STIR_REGISTRIES ${INSTALLED_STIR_REGISTRIES})

CONFIGURE_PACKAGE_CONFIG_FILE(
  src/cmake/STIRConfig.cmake.in
  "${CMAKE_BINARY_DIR}/STIRConfig.cmake"
   INSTALL_DESTINATION "${ConfigPackageLocation}"
   PATH_VARS STIR_INCLUDE_DIRS STIR_REGISTRIES # relocatable variables
)

# create and install STIRTargets*.cmake for the installation-tree
# Note: we cannot have "NAMESPACE stir::" as we would have 
# to prefix all libraries in STIR_LIBRARIES somehow.

install(EXPORT STIRTargets
  DESTINATION "${ConfigPackageLocation}"
)

# install STIRConfig*.cmake
install(
  FILES
    "${CMAKE_BINARY_DIR}/STIRConfig.cmake"
    "${CMAKE_BINARY_DIR}/STIRConfigVersion.cmake"
  DESTINATION "${ConfigPackageLocation}"
)

# install our own Find* cmake files
install(
  DIRECTORY
    "${PROJECT_SOURCE_DIR}/src/cmake/"
  DESTINATION "${ConfigPackageLocation}"
    FILES_MATCHING PATTERN "Find*"
)

#  COMPONENT
#    Devel
