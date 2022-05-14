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
${PROJECT_SOURCE_DIR}/src/buildblock/buildblock_registries.cxx
${PROJECT_SOURCE_DIR}/src/data_buildblock/data_buildblock_registries.cxx
${PROJECT_SOURCE_DIR}/src/IO/IO_registries.cxx
${PROJECT_SOURCE_DIR}/src/recon_buildblock/recon_buildblock_registries.cxx
${PROJECT_SOURCE_DIR}/src/Shape_buildblock/Shape_buildblock_registries.cxx
${PROJECT_SOURCE_DIR}/src/modelling_buildblock/modelling_registries.cxx
${PROJECT_SOURCE_DIR}/src/spatial_transformation_buildblock/spatial_transformation_registries.cxx
${PROJECT_SOURCE_DIR}/src/scatter_buildblock/scatter_registries.cxx
)

SET( STIR_LIBRARIES analytic_FBP3DRP analytic_FBP2D       iterative_OSMAPOSL   iterative_KOSMAPOSL
     iterative_OSSPS
      scatter_buildblock modelling_buildblock listmode_buildblock recon_buildblock  
      display  IO  data_buildblock numerics_buildblock  buildblock 
      spatial_transformation_buildblock
      Shape_buildblock eval_buildblock 
      # repeat for linking
      numerics_buildblock modelling_buildblock listmode_buildblock
      IO modelling_buildblock IO buildblock
)

#copy to PARENT_SCOPE
SET( STIR_REGISTRIES ${STIR_REGISTRIES} PARENT_SCOPE)
SET( STIR_LIBRARIES ${STIR_LIBRARIES} PARENT_SCOPE)

SET( STIR_DIRS
     buildblock
     numerics_buildblock 
     data_buildblock 
     display 
     recon_buildblock 
     modelling_buildblock 
     listmode_buildblock 
     IO 
     spatial_transformation_buildblock
     Shape_buildblock 
     eval_buildblock 
     scatter_buildblock
     utilities 
     scatter_utilities
     modelling_utilities
     listmode_utilities
     analytic/FBP2D
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


SET( STIR_TEST_DIRS
     recon_test  
     test 
     test/numerics
     test/modelling
)

if (STIR_WITH_NiftyPET_PROJECTOR)
  list(APPEND STIR_TEST_DIRS test/NiftyPET_projector)
endif()