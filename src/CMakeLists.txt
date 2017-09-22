# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2012 Kris Thielemans
# Copyright 2016 ETH Zurich
# Copyright 2013, 2014, 2016, 2017 University College London

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
  include_directories(${LLN_INCLUDE_DIRS})
else()
  message(STATUS "ECAT support disabled.")
endif()

if (CERN_ROOT_FOUND)
        set(HAVE_CERN_ROOT ON)
        message(STATUS "CERN ROOT support enabled.")
        include_directories(${CERN_ROOT_INCLUDE_DIRS})
else()
    message(STATUS "CERN ROOT support disabled.")
endif()

if (AVW_FOUND)
  set(HAVE_AVW ON)
  message(STATUS "AVW library IO added.")
  include_directories(${AVW_INCLUDE_DIRS})
else()
  message(STATUS "AVW IO library not used.")
endif()

if (RDF_FOUND)
  set(HAVE_RDF ON)
  message(STATUS "GE RDF support enabled.")
  include_directories(${RDF_INCLUDE_DIRS})

else()
  message(STATUS "RDF support disabled.")
endif()


if (ITK_FOUND) 
  message(STATUS "ITK libraries added.")
  set(HAVE_ITK ON)
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
# With default setting, this will #define USE_PMRT (see STIRConfig.h.in)

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
  add_definitions(${MPI_CXX_COMPILE_FLAGS} -DMPICH_IGNORE_CXX_SEEK)
  # However, currently we decide to add  MPI_CXX_LINK_FLAGS only to those programs that need it,
  # just in case these flags would make normal executables unexecutable without mpi-specific things
  option(STIR_MPI_TIMINGS "Enable timings for MPI code" OFF)
  if (STIR_MPI_TIMINGS)
    add_definitions(-DSTIR_MPI_TIMINGS)
  endif()
endif()

if(STIR_OPENMP)
  find_package(OpenMP REQUIRED)  
  add_definitions(${OpenMP_CXX_FLAGS})
  if ("${OpenMP_C_FLAGS}" STREQUAL "-fopenmp")
    # work-around cmake bug for gcc
    set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS})
  endif ()

endif()

#### Flags for compatibility between different systems
include(CheckFunctionExists)

check_function_exists(getopt HAVE_SYSTEM_GETOPT)
# always include stir/getopt.h for where a system getopt does not exist.
# we provide a replacement in buildblock

# Check for CXX11 smart pointer support.
# This is far more complicated than it should be, largely because we want to support
# older compilers (some claim to be C++-11 but are do not have std::unique_ptr for instance).
# So, we will check BOOST_NO_CXX11_SMART_PTR using check_cxx_symbol_exists.
# However, we need to do that with the same compilers flags as used for STIR.
#
# Sadly, older versions of CMake failed to take CMAKE_CXX_STANDARD into account, so
# we have to do this ourselves.
# Finally, we need to cope with the case where a user passes -std=cxx11 by hand to CMAKE_CXX_FLAGS.
if (CMAKE_VERSION VERSION_LESS 3.8 AND CMAKE_CXX_STANDARD)
  set(CMAKE_REQUIRED_DEFINITIONS -std=cxx${CMAKE_CXX_STANDARD} ${CMAKE_CXX_FLAGS})
else()
  # set policy 0067 to let try_compile honor CMAKE_CXX_STANDARD etc
  # Note: need to set this before including CheckCXXSymbolExists
  cmake_policy(SET CMP0067 NEW)
  set(CMAKE_REQUIRED_DEFINITIONS ${CMAKE_CXX_FLAGS})
endif()
set(CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})
#set(CMAKE_REQUIRED_QUIET FALSE)
include(CheckCXXSymbolExists)
# Finally, do the check!
check_cxx_symbol_exists(BOOST_NO_CXX11_SMART_PTR boost/config.hpp HAVE_NO_CXX11_SMART_PTR)

# Now we introduce 2 options such that the user can override the default behaviour
# (not recommended! unless the above failed of course)
option(STIR_NO_UNIQUE_PTR
       "Disable use of std::unique_ptr" ${HAVE_NO_CXX11_SMART_PTR})
option(STIR_USE_BOOST_SHARED_PTR
       "Use of boost::shared_ptr as opposed to std::shared_ptr" ${HAVE_NO_CXX11_SMART_PTR})
mark_as_advanced(STIR_NO_UNIQUE_PTR STIR_USE_BOOST_SHARED_PTR)

#### Create stir/config.h

# This has to be written where we build somewhere. We will put it somewhere "natural"
# (even though there are no other include files there).
set(CONF_INCLUDE_DIR "${CMAKE_BINARY_DIR}/src/include")
# add it to the include path
include_directories("${CONF_INCLUDE_DIR}")
# create file
configure_file(
  cmake/STIRConfig.h.in
  "${CONF_INCLUDE_DIR}/stir/config.h"
  )
# add it to the install target
install(FILES "${CONF_INCLUDE_DIR}/stir/config.h" DESTINATION include/stir)

#### install include files
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/include/" DESTINATION include)

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


