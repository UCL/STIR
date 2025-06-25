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

include(stir_lib_target)

target_link_libraries(${dir} PUBLIC buildblock numerics_buildblock PRIVATE fmt)
