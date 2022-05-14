# Copyright 2020-2022 University of Pennsylvania
# @Author Nikos Efthimiou
#
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

include(FindPackageHandleStandardArgs)

set(package UPENN)

if (NOT ${package}_PATH)
  set(${package}_PATH "" CACHE PATH "Path to ${package} installation")
endif()

###############################################################
#  Headers
###############################################################

# Find the base folder containing def.h
find_path(${package}_INCLUDE_DIR
  PATHS ${${package}_PATH})
mark_as_advanced(${package}_INCLUDE_DIR)

###############################################################
#  Libraries
###############################################################

set(${package}_required_libraries
  libfit
	libdist
	libgeom
  liblor
  liblist
  libmhdr
  libimagio
  libimagio++
  )
set(${package}_libraries "")

foreach(l ${${package}_required_libraries})
  find_library(${package}_${l}
    NAMES ${l}${CMAKE_SHARED_LIBRARY_SUFFIX} ${l}${CMAKE_STATIC_LIBRARY_SUFFIX}
    PATHS ${${package}_PATH}
  )
  mark_as_advanced(${package}_${l})
  list(APPEND ${package}_libraries ${package}_${l})
endforeach()

find_package_handle_standard_args(${package}
  FOUND_VAR ${package}_FOUND
  REQUIRED_VARS
    ${${package}_libraries} ${package}_INCLUDE_DIR
)
