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

# add project source to cmake path such that it can use our find_package modules and .cmake files
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/cmake)
include(src/cmake/SetC++Version.cmake)

UseCXX("${CMAKE_CXX_STANDARD}")

# set default build-type to Release
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "type of build: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if((${CMAKE_CXX_COMPILER_ID} MATCHES "AppleClang") OR
   (APPLE AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND
    ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

####### Set Version number etc
set(VERSION_MAJOR   3   CACHE STRING "Project major version number.")
set(VERSION_MINOR   1   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
set(VERSION   030100   CACHE STRING "Project version number.")
mark_as_advanced(VERSION VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

set(STIR_VERSION
  ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

####### External packages
# Note: we need to have the find_package statements in the top-level CMakeLists.txt
# such that we can use it in STIRConfig.cmake.in (see below).

#### we need the boost library from boost.org
set(BOOST_ROOT CACHE PATH "root of Boost")
find_package( Boost 1.36.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

#### optional external libraries. 
# Listed here such that we know if we should compile extra utilities
option(DISABLE_LLN_MATRIX "disable use of LLN library" OFF)
option(DISABLE_ITK "disable use of ITK library" OFF)
option(DISABLE_AVW "disable use of AVW library" OFF)
option(DISABLE_RDF "disable use of GE RDF library" OFF)
option(DISABLE_STIR_LOCAL "disable use of LOCAL extensions to STIR" OFF)
option(DISABLE_CERN_ROOT "disable use of Cern ROOT libraries" OFF)
option(STIR_ENABLE_EXPERIMENTAL "disable use of STIR experimental code" OFF) # disable by default

if(NOT DISABLE_ITK)
   # See if we can find a compiled version of ITK (http://www.itk.org/)
   find_package(ITK QUIET)
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

set(ConfigPackageLocation lib/cmake/)

# Set STIR_INCLUDE_DIRS before exporting such that it will refer to
# the installed files, not the source.
set (STIR_INCLUDE_DIRS "include")

CONFIGURE_PACKAGE_CONFIG_FILE(
  src/cmake/STIRConfig.cmake.in
  "${CMAKE_BINARY_DIR}/STIRConfig.cmake"
   INSTALL_DESTINATION "${ConfigPackageLocation}"
   PATH_VARS STIR_INCLUDE_DIRS
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
