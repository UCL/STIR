
PROJECT(STIR)

cmake_minimum_required(VERSION 2.6)

# add project source to cmake path such that it can use our find_package modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

INCLUDE(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)

option(STIR_MPI 
       "Compile with MPI" OFF)
option(STIR_OPENMP 
       "Compile with OpenMP" OFF)

# we need the boost library from boost.org
set(BOOST_ROOT CACHE PATH "root of Boost")
find_package( Boost 1.36.0 REQUIRED )

##### optional external libraries. 
# Listed here such that we know if we should compile extra utilities
option(DISABLE_LLN_MATRIX "disable use of LLN library" OFF)
option(DISABLE_AVW "disable use of AVW library" OFF)

if(NOT DISABLE_LLN_MATRIX)
  find_package(LLN)
endif()

if(NOT DISABLE_AVW)
  find_package(AVW)
endif()

if (LLN_FOUND)
  set(HAVE_ECAT ON)
  message(STATUS "ECAT support enabled.")
  add_definitions(-D HAVE_LLN_MATRIX)
  include_directories(${LLN_INCLUDE_DIRS})

else()
  message(STATUS "ECAT support disabled.")
endif()

if (AVW_FOUND)
  message(STATUS "AVW library IO added.")
  add_definitions(-D HAVE_AVW)
  include_directories(${AVW_INCLUDE_DIRS})
else()
  message(STATUS "AVW IO library not used.")
endif()

#### Compiler specific flags for fastest execution etc

# gcc specific stuff
if (CMAKE_COMPILER_IS_GNUCC)
     add_definitions(-Wall -Wno-deprecated)
     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -malign-double")
endif ()

# enable Intel compiler specific flags
if ($(CMAKE_C_COMPILER) STREQUAL "icc")
     list(APPEND CMAKE_CXX_FLAGS_RELEASE -fast)
endif()

#### Flags for parallel execution of certain STIR modules

if (STIR_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  list(PREPEND STIR_LIBRARIES ${MPI_LIBRARIES})
  # not sure if we need to use MPI_LINK_FLAGS
  add_definitions(${MPI_COMPILE_FLAGS} -DMPICH_IGNORE_CXX_SEEK -DSTIR_MPI)
  option(STIR_MPI_TIMINGS, "Enable timings for MPI code", OFF)
  if (STIR_MPI_TIMINGS)
    add_definitions(-DSTIR_MPI_TIMINGS)
  endif()
endif()

if(STIR_OPENMP)
  find_package(OpenMP REQUIRED)  
  add_definitions(${OpenMP_CXX_FLAGS} -DSTIR_OPENMP)
endif()

##### STIR library

# add the STIR include directory to the search path for the compiler
include_directories ("${PROJECT_SOURCE_DIR}/include")

SET ( STIR_REGISTRIES
${CMAKE_HOME_DIRECTORY}/buildblock/buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/data_buildblock/data_buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/IO/IO_registries.cxx
${CMAKE_HOME_DIRECTORY}/recon_buildblock/recon_buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/Shape_buildblock/Shape_buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/modelling_buildblock/modelling_registries.cxx
)

SET( STIR_LIBRARIES analytic_FBP3DRP analytic_FBP2D       iterative_OSMAPOSL  
     iterative_OSSPS
      modelling_buildblock listmode_buildblock recon_buildblock  
      display  IO  data_buildblock numerics_buildblock  buildblock 
      Shape_buildblock eval_buildblock 
      # repeat for linking
      numerics_buildblock modelling_buildblock listmode_buildblock)


SET( STIR_DIRS
     buildblock
     numerics_buildblock 
     data_buildblock 
     recon_buildblock 
     modelling_buildblock 
     listmode_buildblock 
     display 
     IO 
     Shape_buildblock 
     eval_buildblock 
     scatter_buildblock
     utilities 
     listmode_utilities
     analytic/FBP2D
     analytic/FBP3DRP
     iterative/OSMAPOSL  
     iterative/OSSPS
)

#        recon_test  test test/numerics)

if (HAVE_ECAT)
  list(APPEND STIR_DIRS utilities/ecat)
endif()

# go and look for CMakeLists.txt files in all those directories
foreach(STIR_DIR ${STIR_DIRS})
	ADD_SUBDIRECTORY(${STIR_DIR})
#        INCLUDE(${STIR_DIR}/lib.cmake)
endforeach(STIR_DIR)


#TARGET_LINK_LIBRARIES(STIR recon_buildblock) #${STIR_LIBRARIES})

# add the examples directory as well, but don't build it by default
ADD_SUBDIRECTORY(examples EXCLUDE_FROM_ALL)


# see if there's a local subdirectory.
find_path(STIR_LOCAL NAME "CMakeLists.txt" PATHS "${PROJECT_SOURCE_DIR}/local/" NO_DEFAULT_PATH)
if (STIR_LOCAL)
  add_subdirectory(${STIR_LOCAL})
endif()


### Export settings such that another project can import them.
# Save the compiler settings 
INCLUDE(${CMAKE_ROOT}/Modules/CMakeExportBuildSettings.cmake)
CMAKE_EXPORT_BUILD_SETTINGS(${STIR_BINARY_DIR}/STIRBuildSettings.cmake)

# Save library dependencies.
EXPORT_LIBRARY_DEPENDENCIES(${STIR_BINARY_DIR}/STIRLibraryDepends.cmake)
