#
#
# Copyright 2013 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

set(dir spatial_transformation_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
   SpatialTransformation.cxx
   GatedSpatialTransformation.cxx
   warp_image.cxx
   InvertAxis.cxx
) 

set(TARGET ${STIR_SPATIAL_TRANSFORMATION_BUIDBLOCK_LIB})
target_sources(${TARGET} PRIVATE ${${dir_LIB_SOURCES}})
