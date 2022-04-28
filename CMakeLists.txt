# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
cmake_minimum_required(VERSION 3.1.0)


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

if (CMAKE_CXX_COMPILER_ID)
   if((${CMAKE_CXX_COMPILER_ID} MATCHES "AppleClang") OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
      # Ignore undefined template var warning (see https://github.com/UCL/STIR/issues/126)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")
      if (APPLE)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
      endif()
   endif()
else()
   message(WARNING "CMAKE_CXX_COMPILER_ID is not set. We are therefore not able to set some options.")
endif()

####### Set Version number etc
set(VERSION_MAJOR   5)
set(VERSION_MINOR   1)
set(VERSION_PATCH   0)
set(VERSION   050100) # only used in STIRConfig.h.in

set(STIR_VERSION
  ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

####### Installation directories
# Use BeOS locations as in
# https://gitlab.kitware.com/cmake/cmake/blob/master/Source/CMakeInstallDestinations.cmake
# However, use CYGWIN location also elsewhere (as that's what's used in Ubuntu (and others?))
if(BEOS)
  set(STIR_DOC_DIR "documentation/doc/stir-${VERSION_MAJOR}.${VERSION_MINOR}")
else() #if(CYGWIN)
  set(STIR_DOC_DIR "share/doc/stir-${VERSION_MAJOR}.${VERSION_MINOR}")
#else()
#  set(STIR_DOC_DIR "doc/stir-${VERSION_MAJOR}.${VERSION_MINOR}")
endif()
# TODO append version numbers
set(STIR_DATA_DIR "share/stir") 
set(ConfigPackageLocation lib/cmake/)
set(STIR_CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/stir/config" CACHE PATH "") 
message("${STIR_CONFIG_DIR} defined")
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
option(DISABLE_HDF5 "disable use of HDF5 libraries" OFF)
option(DISABLE_STIR_LOCAL "disable use of LOCAL extensions to STIR" OFF)
option(DISABLE_CERN_ROOT "disable use of Cern ROOT libraries" OFF)
option(DISABLE_NLOHMANN_JSON "disable use of nlohmann JSON libraries" OFF)
option(STIR_ENABLE_EXPERIMENTAL "disable use of STIR experimental code" OFF) # disable by default
option(DISABLE_NiftyPET_PROJECTOR "disable use of NiftyPET projector" OFF)
option(DISABLE_Parallelproj_PROJECTOR "disable use of Parallelproj projector" OFF)
OPTION(DOWNLOAD_ZENODO_TEST_DATA "download zenodo data for tests" OFF)
option(DISABLE_UPENN "disable use of UPENN filetypes" OFF)


if(NOT DISABLE_ITK)
   # See if we can find a compiled version of ITK (http://www.itk.org/)
   find_package(ITK 4.9)
   if (ITK_FOUND)
      include(${ITK_USE_FILE})
      set(HAVE_ITK ON)
   endif()
endif()

if(NOT DISABLE_LLN_MATRIX)
  find_package(LLN)
endif()

if(NOT DISABLE_CERN_ROOT)
  find_package(CERN_ROOT)
  if (CERN_ROOT_FOUND)
    message(STATUS "ROOT Version: ${CERN_ROOT_VERSION}")
    if (${CERN_ROOT_VERSION} VERSION_GREATER 6.23.99)
        message(STATUS "ROOT Version is >= 6.24. Setting the minimum CXX version to 14.")
        UseCXX(14)
    endif()
  endif()
endif()

if(NOT DISABLE_AVW)
  find_package(AVW)
endif()

if(NOT DISABLE_RDF)
  find_package(RDF)
endif()

if(NOT DISABLE_HDF5)
  find_package(HDF5 COMPONENTS CXX)
endif()

if(NOT DISABLE_NLOHMANN_JSON)
    find_package(nlohmann_json 3.2.0)# QUIET)
    if (nlohmann_json_FOUND)
      message(STATUS "nlohmann JSON library found.")
      set(HAVE_JSON ON)
    else()
      message(STATUS "nlohmann JSON library not found. Try and set nlohmann_json_DIR if you have it.")
    endif()
endif()

if(NOT DISABLE_UPENN)
	find_package(JANSSON)
		if(JANSSON_FOUND)
			find_package(ZLIB)
			if(ZLIB_FOUND)
				find_package(UPENN) 
				if (UPENN_FOUND)
					message(STATUS "UPENN libary found.")
					#set(HAVE_UPENN ON)
				else()
					message(STATUS "UPENN library not found. Try and set the DIR manually.")
				endif()
			else()
				message(STATUS "UPENN library needs Zlib")
			endif()
		else()
			message(STATUS "UPENN library needs Jansson")
		endif()
endif()

# Legacy options
option(STIR_PROJECTORS_AS_V3 "Enable STIR version 3 legacy code" OFF)
option(STIR_LEGACY_IGNORE_VIEW_OFFSET "Ignore using scanner view-offset (or intrinsic azimuthal tilt), as in STIR 4 or earlier" OFF)
option(STIR_ROOT_ROTATION_AS_V4 "Ignore the changes to the ROOT rotation introduced in STIR version 5" OFF)

# NiftyPET projector
if(NOT DISABLE_NiftyPET_PROJECTOR OR NOT DISABLE_Parallelproj_PROJECTOR)
  find_package(CUDAToolkit QUIET)
endif()

if(NOT DISABLE_NiftyPET_PROJECTOR)
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

# Parallelproj
if(NOT DISABLE_Parallelproj_PROJECTOR)
    find_package(parallelproj 0.8)
    if (parallelproj_FOUND)
      set(STIR_WITH_Parallelproj_PROJECTOR ON)
      if (parallelproj_built_with_CUDA)
        message(STATUS "Found parallelproj (will use its CUDA support)")
      else()
        message(STATUS "Found parallelproj (but using its OpenMP version as it wasn't built with CUDA)")
      endif()
    endif()
endif()
if (STIR_WITH_Parallelproj_PROJECTOR)
  message(STATUS "Parallelproj projector support enabled.")
else()
  message(STATUS "Parallelproj projector support disabled or not available.")
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

# install examples
install(
  DIRECTORY
    examples
  COMPONENT DOC
  DESTINATION "${STIR_DOC_DIR}"
   FILES_MATCHING
    PATTERN *
    PATTERN my_* EXCLUDE
    PATTERN *~  EXCLUDE
    PATTERN *.bak EXCLUDE
    PATTERN  *.log EXCLUDE
)

# install documentation
install(
  DIRECTORY
    documentation
  COMPONENT DOC
    DESTINATION "${STIR_DOC_DIR}"
  FILES_MATCHING
    PATTERN *
    PATTERN *.aux EXCLUDE
    PATTERN *.dvi EXCLUDE
    PATTERN *.out EXCLUDE
    PATTERN *.toc EXCLUDE
    PATTERN *~  EXCLUDE
    PATTERN *.bak EXCLUDE
    PATTERN  *.log EXCLUDE
)
#  COMPONENT
#    Devel
