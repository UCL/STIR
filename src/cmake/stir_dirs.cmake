#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file for building STIR. 
# This file is included by CMakeLists.txt and sets variables
# listing all directories. These variables are then used in CMakeLists.txt.



# add the STIR include directory to the search path for the compiler
SET (STIR_INCLUDE_DIR 
     "${PROJECT_SOURCE_DIR}/src/include"
)

# add STIR include directories before existing include paths such that
# files there are used, as opposed to an existing STIR installation elsewhere
include_directories (BEFORE "${STIR_INCLUDE_DIR}")

# registries
SET (STIR_IO_REGISTRIES
     ${PROJECT_SOURCE_DIR}/src/IO/IO_registries.cxx
     )

SET ( STIR_REGISTRIES
${PROJECT_SOURCE_DIR}/src/IO/IO_registries.cxx
)

SET( STIR_DIRS
     buildblock
     numerics_buildblock 
     data_buildblock 
     recon_buildblock 
     IO
     utilities
     )

SET( STIR_TEST_DIRS
  test 
)

if (NOT MINI_STIR)
  list(APPEND STIR_REGISTRIES
    ${PROJECT_SOURCE_DIR}/src/buildblock/buildblock_registries.cxx
    ${PROJECT_SOURCE_DIR}/src/recon_buildblock/recon_buildblock_registries.cxx
    ${PROJECT_SOURCE_DIR}/src/data_buildblock/data_buildblock_registries.cxx
    ${PROJECT_SOURCE_DIR}/src/Shape_buildblock/Shape_buildblock_registries.cxx
    ${PROJECT_SOURCE_DIR}/src/modelling_buildblock/modelling_registries.cxx
    ${PROJECT_SOURCE_DIR}/src/spatial_transformation_buildblock/spatial_transformation_registries.cxx
    ${PROJECT_SOURCE_DIR}/src/scatter_buildblock/scatter_registries.cxx
    )

  # need to list IO first such that its dependencies on other libraries are resolved
  SET( STIR_LIBRARIES IO analytic_FBP3DRP analytic_FBP2D analytic_SRT2D analytic_SRT2DSPECT  iterative_OSMAPOSL   iterative_KOSMAPOSL
    iterative_OSSPS
    scatter_buildblock modelling_buildblock listmode_buildblock recon_buildblock  
    display   data_buildblock numerics_buildblock buildblock
    spatial_transformation_buildblock
    Shape_buildblock eval_buildblock 
    )

  list(APPEND STIR_DIRS
    display 
    modelling_buildblock 
    listmode_buildblock 
    spatial_transformation_buildblock
    Shape_buildblock 
    eval_buildblock 
    scatter_buildblock
    scatter_utilities
    modelling_utilities
    listmode_utilities
    analytic/FBP2D
    analytic/SRT2D
    analytic/SRT2DSPECT
    analytic/FBP3DRP
    iterative/OSMAPOSL  
    iterative/KOSMAPOSL
    iterative/OSSPS
    iterative/POSMAPOSL  
    iterative/POSSPS
    SimSET
    SimSET/scripts
    )

  if (HAVE_ECAT)
    list(APPEND STIR_DIRS utilities/ecat)
  endif()

  if (HAVE_UPENN)
    list(APPEND STIR_DIRS utilities/UPENN)
  endif()

  
  list(APPEND STIR_TEST_DIRS
    test/numerics
    test/modelling
    )

  if (STIR_WITH_NiftyPET_PROJECTOR)
    list(APPEND STIR_TEST_DIRS test/NiftyPET_projector)
  endif()

else() # MINI_STIR

  SET( STIR_LIBRARIES IO recon_buildblock  
      data_buildblock numerics_buildblock buildblock      
      )

endif()

#copy to PARENT_SCOPE
SET( STIR_REGISTRIES ${STIR_REGISTRIES} PARENT_SCOPE)
SET( STIR_LIBRARIES ${STIR_LIBRARIES} PARENT_SCOPE)
