# Copyright 2019-2020 University College London
# @Author Richard Brown
#
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

include(FindPackageHandleStandardArgs)

set(package NiftyPET)

if (NOT ${package}_PATH)
  set(${package}_PATH "" CACHE PATH "Path to ${package} installation")
endif()

###############################################################
#  Headers
###############################################################

# Find the base folder containing def.h
find_path(${package}_INCLUDE_DIR "niftypet/nipet/def.h"
  PATHS ${${package}_PATH}
  PATH_SUFFIXES include
  )
mark_as_advanced(${package}_INCLUDE_DIR)

###############################################################
#  Libraries
###############################################################

set(${package}_required_libraries
  petprj
  mmr_auxe
  mmr_lmproc
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

###############################################################
#  Check everything has been found
###############################################################

find_package_handle_standard_args(${package}
  FOUND_VAR ${package}_FOUND
  REQUIRED_VARS
    ${${package}_libraries} ${package}_INCLUDE_DIR
)

###############################################################
#  Get header subdirectories
###############################################################

SET(${package}_INCLUDE_DIRS
  "${${package}_INCLUDE_DIR}/niftypet/nipet"
  "${${package}_INCLUDE_DIR}/niftypet/nipet/dinf"
  "${${package}_INCLUDE_DIR}/niftypet/nipet/lm/src"
  "${${package}_INCLUDE_DIR}/niftypet/nipet/prj/src"
  "${${package}_INCLUDE_DIR}/niftypet/nipet/sct/src"
  "${${package}_INCLUDE_DIR}/niftypet/nipet/src"
)

###############################################################
#  Create imported targets
###############################################################

foreach(l ${${package}_required_libraries})
  add_library(${package}::${l} UNKNOWN IMPORTED)
  set_target_properties(${package}::${l} PROPERTIES
    IMPORTED_LOCATION "${${package}_${l}}"
    INTERFACE_INCLUDE_DIRECTORIES "${${package}_INCLUDE_DIRS}"
  )
endforeach()
