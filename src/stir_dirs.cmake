#
# $Id$
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - $Date$ Kris Thielemans

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

# cmake helper file for building STIR. 
# This file is included by CMakeLists.txt and sets variables
# listing all directories. These variables are then used in CMakeLists.txt.



# add the STIR include directory to the search path for the compiler
include_directories ("${PROJECT_SOURCE_DIR}/include")

SET ( STIR_REGISTRIES
${CMAKE_HOME_DIRECTORY}/buildblock/buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/data_buildblock/data_buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/IO/IO_registries.cxx
${CMAKE_HOME_DIRECTORY}/recon_buildblock/recon_buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/Shape_buildblock/Shape_buildblock_registries.cxx
${CMAKE_HOME_DIRECTORY}/modelling_buildblock/modelling_registries.cxx
${CMAKE_HOME_DIRECTORY}/spatial_transformation_buildblock/spatial_transformation_registries.cxx
)

SET( STIR_LIBRARIES analytic_FBP3DRP analytic_FBP2D       iterative_OSMAPOSL  
     iterative_OSSPS
      scatter_buildblock modelling_buildblock listmode_buildblock recon_buildblock  
      display  IO  data_buildblock numerics_buildblock  buildblock 
      spatial_transformation_buildblock
      Shape_buildblock eval_buildblock 
      # repeat for linking
      numerics_buildblock modelling_buildblock listmode_buildblock)


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
     iterative/OSSPS
     iterative/POSMAPOSL  
     iterative/POSSPS
     scripts
     SimSET
     SimSET/scripts
)


if (HAVE_ECAT)
  list(APPEND STIR_DIRS utilities/ecat)
endif()


SET( STIR_TEST_DIRS
     recon_test  
     test 
     test/numerics
     test/modelling
)
