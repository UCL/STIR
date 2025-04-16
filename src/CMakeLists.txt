# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2012 Kris Thielemans
# Copyright 2016 ETH Zurich
# Copyright 2013, 2014, 2016-2023 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

option(MINI_STIR
       "Create a mini version of STIR with only basic functionality (for testing only)" OFF)
mark_as_advanced(MINI_STIR)
     
option(STIR_MPI 
       "Compile with MPI" OFF)
option(STIR_OPENMP 
       "Compile with OpenMP" ON)

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
else()
  message(STATUS "ECAT support disabled.")
endif()

if (CERN_ROOT_FOUND)
        set(HAVE_CERN_ROOT ON)
        message(STATUS "CERN ROOT support enabled.")
else()
    message(STATUS "CERN ROOT support disabled.")
endif()

if (UPENN_FOUND)
	set(HAVE_UPENN ON)
	message(STATUS "UPENN libs support enabled.")
else()
	message(STATUS "UPENN libs support disabled.")
endif()

if ((NOT DISABLE_HDF5) AND HDF5_FOUND)
  set(HAVE_HDF5 ON)
  message(STATUS "HDF5 support enabled.")
else()
  message(STATUS "HDF5 support disabled.")
endif()

if ((NOT DISABLE_ITK) AND ITK_FOUND) 
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
  # require 3.3 for descent FindMatlab.cmake
  cmake_minimum_required(VERSION 3.3.0)
  set(CMAKE_POSITION_INDEPENDENT_CODE True)
  message(STATUS "Trying to find Matlab. This might take a while. If this fails, set Matlab_ROOT_DIR.")
  FIND_PACKAGE(Matlab COMPONENTS MX_LIBRARY REQUIRED)
  matlab_get_mex_suffix("${Matlab_ROOT_DIR}" Matlab_MEX_EXT)
  message(STATUS "Found Matlab_LIBRARIES: ${Matlab_LIBRARIES}")
  message(STATUS "Found Matlab_INCLUDE_DIRS: ${Matlab_INCLUDE_DIRS}")
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

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # work around https://gitlab.kitware.com/cmake/cmake/-/issues/26037
    set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX -latomic)
    message(STATUS "OpenMP Linker flags for Clang: ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
  endif()
endif()

#### Flags for compatibility between different systems
include(CheckFunctionExists)

check_function_exists(getopt HAVE_SYSTEM_GETOPT)
# always include stir/getopt.h for where a system getopt does not exist.
# we provide a replacement in buildblock

# Check for CXX11 smart pointer support.
# This is far more complicated than it should be, largely because we want to support
# older compilers (some claim to be C++-11 but are do not have std::unique_ptr for instance).
# However, we need to do that with the same compilers flags as used for STIR.
#
# Sadly, older versions of CMake failed to take CMAKE_CXX_STANDARD into account, so
# we have to do this ourselves.
# Finally, we need to cope with the case where a user passes -std=cxx11 by hand to CMAKE_CXX_FLAGS.
set(CMAKE_REQUIRED_DEFINITIONS ${CMAKE_CXX_FLAGS})
# set policy 0067 to let try_compile honor CMAKE_CXX_STANDARD etc
# Note: need to set this before including CheckCXXSymbolExists
cmake_policy(SET CMP0067 NEW)
#set(CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})
#set(CMAKE_REQUIRED_QUIET FALSE)
# Finally, do the checks!

include(CheckCXXSourceCompiles)
# Perform test to see if std::unique_ptr exists
check_cxx_source_compiles("
   #include <memory>
   std::unique_ptr<int> f(){int *ptr = new int;return std::unique_ptr<int>(ptr);}
   int main(){auto p = f(); return 0;}
   "
        HAVE_STD_UNIQUE_PTR)

if(NOT HAVE_STD_UNIQUE_PTR)
    message(FATAL_ERROR "Problems compiling with std::unique_ptr. Giving up.")
    set(STIR_NO_UNIQUE_PTR ON)
endif()

# Perform test to see if std::shared_ptr exists
check_cxx_source_compiles("
        #include <memory>
        std::shared_ptr<int> f(){int *ptr = new int; return std::shared_ptr<int>(ptr);}
        int main(){auto p = f(); return 0;}
    "
       HAVE_STD_SHARED_PTR
        )

# Byte order
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    # Use CMake to find byte order of target system.
    # We create some variables that are then used in config.h.in
    if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
        set(BIG_ENDIAN_BYTE_ORDER_FROM_CMAKE TRUE)
        message(STATUS "CMake detected a big_endian system")
    elseif (CMAKE_CXX_BYTE_ORDER STREQUAL "LITTLE_ENDIAN")
        set(LITTLE_ENDIAN_BYTE_ORDER_FROM_CMAKE TRUE)
        message(STATUS "CMake detected a little_endian system")
    endif()
endif()

#### Create stir/config.h

# This has to be written where we build somewhere. We will put it somewhere "natural"
# (even though there are no other include files there).
set(CONF_INCLUDE_DIR "${CMAKE_BINARY_DIR}/src/include")
# add it to the include path. Make sure we prepend it.
include_directories(BEFORE "${CONF_INCLUDE_DIR}")
# create file
configure_file(
  cmake/STIRConfig.h.in
  "${CONF_INCLUDE_DIR}/stir/config.h"
  )
# add it to the install target
install(FILES "${CONF_INCLUDE_DIR}/stir/config.h" DESTINATION "${STIR_INCLUDE_INSTALL_DIR}/stir")

#### install include files
set (INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/include")
install(DIRECTORY "${INCLUDE_DIR}/" DESTINATION "${STIR_INCLUDE_INSTALL_DIR}")

#### find header files for re-use in stir_exe_targets.cmake and doxygen target
include(FindAllHeaderFiles)
find_all_header_files(ALL_HEADERS ALL_INLINES ALL_TXXS "${INCLUDE_DIR}/stir")

#### STIR library
include(stir_dirs)

# see if there's a local subdirectory.
if(NOT DISABLE_STIR_LOCAL)
  find_path(STIR_LOCAL NAME "extra_stir_dirs.cmake" PATHS "${PROJECT_SOURCE_DIR}/src/local/" NO_DEFAULT_PATH)
  if (STIR_LOCAL)
    # append any headers there
    find_all_header_files(ALL_HEADERS ALL_INLINES ALL_TXXS "${STIR_LOCAL}")
    # include user file
    include(${STIR_LOCAL}/extra_stir_dirs.cmake)
  endif()
endif()

# Include the experimental code
if (STIR_ENABLE_EXPERIMENTAL)
  # append any headers there
  find_all_header_files(ALL_HEADERS ALL_INLINES ALL_TXXS "${INCLUDE_DIR}/stir_experimental")
  # add the sub-dir
  ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/experimental)
endif()

# create an object library with all the "registries"
# Warning: dependencies for object libraries are not transitive!
add_library(stir_registries OBJECT ${STIR_REGISTRIES})
target_link_libraries(stir_registries PRIVATE fmt)
# TODO, really should use stir_libs.cmake
target_include_directories(stir_registries PRIVATE ${STIR_INCLUDE_DIR})
target_include_directories(stir_registries PRIVATE ${Boost_INCLUDE_DIR})

# go and look for CMakeLists.txt files in all those directories
foreach(STIR_DIR ${STIR_DIRS} ${STIR_TEST_DIRS})
	ADD_SUBDIRECTORY(${STIR_DIR})
#        INCLUDE(${STIR_DIR}/lib.cmake)
endforeach(STIR_DIR)


install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/config/ DESTINATION ${STIR_CONFIG_DIR}
                FILES_MATCHING PATTERN "*.json")
message(STATUS "Installing JSON files from ${CMAKE_SOURCE_DIR}/src/config/ to  ${STIR_CONFIG_DIR}")
            
## Build doxygen documentation
# Has to be after previous things such that all variables are filled
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()
endif()

if(BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
    if(DOXYGEN_DOT_FOUND)
        set(HAVE_DOT 1)
        message(STATUS "Graphviz dot found. It will be used for the Doxygen documentation.")
    else()
        set(HAVE_DOT 0)
    	message(WARNING "Graphviz dot is not found. It is recommended to install Graphviz for nicer Doxygen documentation")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    get_property(DOX_STRIP_FROM_INC_PATH DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    string(REPLACE ";" " \\\n\t\t\t\t" DOX_STRIP_FROM_INC_PATH "${DOX_STRIP_FROM_INC_PATH}")

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    # The following lines are modifications of Jannis Fischer's original. The modifications were inspired by
    # https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/

    # Add a custom command that does everything, but also generates a doxygen.stamp file.
    # This should prevent doxygen being re-run when not needed.
    add_custom_command(
        OUTPUT doxygen.stamp
        DEPENDS ${doxyfile} ${ALL_HEADERS}
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        COMMAND ${CMAKE_COMMAND} -E touch doxygen.stamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    # Now add a custom target that depends on the doxygen.stamp.
    # If BUILD_DOCUMENTATION, we add this target to the dependency list of "ALL"
    # to make sure it always gets run.
    # Otheriwse, the user has to build the target manually.
    set(DOXY_ALL ALL)
      
    add_custom_target(
        RUN_DOXYGEN ${DOXY_ALL}
        DEPENDS doxygen.stamp)

    # Doxyfile.in currently tells doxygen to output in PROJECT_BINARY_DIR
    install(DIRECTORY ${PROJECT_BINARY_DIR}/html/
            COMPONENT DOC
            DESTINATION ${STIR_DOC_DIR}/doxygen)
endif()

#### SWIG settings
# Needs to be added after running doxygen as we use its xml output for swig
ADD_SUBDIRECTORY(swig)


