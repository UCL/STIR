# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2012 Kris Thielemans
# Copyright 2013 University College London

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

INCLUDE(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)

option(STIR_MPI 
       "Compile with MPI" OFF)
option(STIR_OPENMP 
       "Compile with OpenMP" OFF)

option(BUILD_TESTING 
       "Build test programs" ON)

option(BUILD_EXECUTABLES
       "Build executables" ON)

option(BUILD_SHARED_LIBS 
       "Use shared libraries" OFF)

### Settings for external libraries

if (LLN_FOUND)
  set(HAVE_ECAT ON)
  message(STATUS "ECAT support enabled.")
  add_definitions(-D HAVE_LLN_MATRIX)
  include_directories(${LLN_INCLUDE_DIRS})

else()
  message(STATUS "ECAT support disabled.")
endif()

if (CERN_ROOT_FOUND)
        set(HAVE_CERN_ROOT ON)
        message(STATUS "CERN ROOT support enabled.")
        add_definitions(-DHAVE_CERN_ROOT)
        include_directories(${CERN_ROOT_INCLUDE_DIRS})
else()
    message(STATUS "CERN ROOT support disabled.")
endif()

if (AVW_FOUND)
  message(STATUS "AVW library IO added.")
  add_definitions(-D HAVE_AVW)
  include_directories(${AVW_INCLUDE_DIRS})
else()
  message(STATUS "AVW IO library not used.")
endif()

if (RDF_FOUND)
  set(HAVE_RDF ON)
  message(STATUS "GE RDF support enabled.")
  add_definitions(-D HAVE_RDF -D STIR_USE_GE_IO)
  include_directories(${RDF_INCLUDE_DIRS})

else()
  message(STATUS "RDF support disabled.")
endif()


if (ITK_FOUND) 
  message(STATUS "ITK libraries added.")
  set(HAVE_ITK ON)
  add_definitions(-DHAVE_ITK)
  include(${ITK_USE_FILE})
else()
  message(STATUS "ITK support disabled.")
endif()

#### Swig related
option(BUILD_SWIG_PYTHON
       "Use SWIG to generate python bindings for STIR (experimental)" OFF)

option(BUILD_SWIG_OCTAVE
       "Use SWIG to generate octave bindings for STIR (experimental)" OFF)

option(BUILD_SWIG_MATLAB
       "Use SWIG to generate matlab bindings for STIR (experimental)" OFF)

# Octave support doesn't really work very well yet, so hide it a bit
mark_as_advanced(BUILD_SWIG_OCTAVE)

if (BUILD_SWIG_MATLAB)
  set(CMAKE_POSITION_INDEPENDENT_CODE True)
  FIND_PACKAGE(MATLAB REQUIRED)
  # Add matlab compiler flags to everything
  # TODO this is probably not necessary anymore with CMAKE_POSITION_INDEPENDENT_CODE set to true
  add_definitions(${MATLAB_CXXFLAGS})

endif()

if (BUILD_SWIG_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE True)
endif()

if (BUILD_SWIG_OCTAVE)
  set(CMAKE_POSITION_INDEPENDENT_CODE True)
endif()

#### Flags for defaults
option(STIR_DEFAULT_PROJECTOR_AS_V2
       "Use same default projectors as STIR 1.x and 2.x (not recommended)" OFF)
#mark_as_advanced(STIR_DEFAULT_PROJECTOR_AS_VERSION_2)

if (STIR_DEFAULT_PROJECTOR_AS_V2)
 # nothing
else()
  add_definitions(-D USE_PMRT)
endif()

#### Flags for compatibility between different systems

check_function_exists(getopt HAVE_SYSTEM_GETOPT)
if (HAVE_SYSTEM_GETOPT)
  add_definitions(-D HAVE_SYSTEM_GETOPT)
  # buildblock will have a replacement for it as long as you include stir/getopt.h
endif()

#### Compiler specific flags for fastest execution etc

# gcc specific stuff
if (CMAKE_COMPILER_IS_GNUCC)
     add_definitions(-Wall -Wno-deprecated)
     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
endif ()

# enable Intel compiler specific flags
if ($(CMAKE_C_COMPILER) STREQUAL "icc")
     list(APPEND CMAKE_CXX_FLAGS_RELEASE -fast)
endif()

#### Flags for parallel execution of certain STIR modules

if (STIR_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})  
  # we add MPI_CXX_COMPILE_FLAGS for all files
  add_definitions(${MPI_CXX_COMPILE_FLAGS} -DMPICH_IGNORE_CXX_SEEK -DSTIR_MPI)
  # However, currently we decide to add  MPI_CXX_LINK_FLAGS only to those programs that need it,
  # just in case these flags would make normal executables unexecutable without mpi-specific things
  option(STIR_MPI_TIMINGS "Enable timings for MPI code" OFF)
  if (STIR_MPI_TIMINGS)
    add_definitions(-DSTIR_MPI_TIMINGS)
  endif()
endif()

if(STIR_OPENMP)
  find_package(OpenMP REQUIRED)  
  add_definitions(${OpenMP_CXX_FLAGS} -DSTIR_OPENMP)
  if ("${OpenMP_C_FLAGS}" STREQUAL "-fopenmp")
    # work-around cmake bug for gcc
    set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS})
  endif ()

endif()

#### STIR library
include(stir_dirs)

# see if there's a local subdirectory.
if(NOT DISABLE_STIR_LOCAL)
  find_path(STIR_LOCAL NAME "extra_stir_dirs.cmake" PATHS "${PROJECT_SOURCE_DIR}/src/local/" NO_DEFAULT_PATH)
  if (STIR_LOCAL)
    include(${STIR_LOCAL}/extra_stir_dirs.cmake)
  endif()
endif()



# go and look for CMakeLists.txt files in all those directories
foreach(STIR_DIR ${STIR_DIRS} ${STIR_TEST_DIRS})
	ADD_SUBDIRECTORY(${STIR_DIR})
#        INCLUDE(${STIR_DIR}/lib.cmake)
endforeach(STIR_DIR)


#### SWIG settings
ADD_SUBDIRECTORY(swig)

#### install include files
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/include/" DESTINATION include)

## CMake variables to build doxygen documentation
## Create html documentation using Doxygen
## Has to be at the end such that all variables are filled
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_DOT_FOUND})
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    if(NOT DOXYGEN_DOT_FOUND)
    	message(FATAL_ERROR "Dot is needed for the Doxygen documentation to be generated.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    get_property(DOX_STRIP_FROM_INC_PATH DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    string(REPLACE ";" " \\\n\t\t\t\t" DOX_STRIP_FROM_INC_PATH "${DOX_STRIP_FROM_INC_PATH}")

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

#    install(DIRECTORY ${PROJECT_BINARY_DIR}/html DESTINATION share/doc/stir)
endif()


