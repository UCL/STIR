# Copyright 2019 - University College London - Richard Brown

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

# Get the petprj and aux libraries
if (NOT EXISTS "${NIFTYPET_SOURCE_DIR}/niftypet/nipet/prj/src/prjf.h")
	MESSAGE(FATAL_ERROR "NiftyPET source directory incorrect 
		(${NIFTYPET_SOURCE_DIR}/niftypet/nipet/prj/src/prjf.h doesn't exist).")
endif()

SET(NIFTYPET_INCLUDE_DIRS 
	"${NIFTYPET_SOURCE_DIR}/niftypet/nipet;"
	"${NIFTYPET_SOURCE_DIR}/niftypet/nipet/dinf;"
	"${NIFTYPET_SOURCE_DIR}/niftypet/nipet/lm/src;"
	"${NIFTYPET_SOURCE_DIR}/niftypet/nipet/prj/src;"
	"${NIFTYPET_SOURCE_DIR}/niftypet/nipet/sct/src;"
	"${NIFTYPET_SOURCE_DIR}/niftypet/nipet/src")
SET(NIFTYPET_LIBRARIES 
	${NIFTYPET_PETPRJ_LIB}
	${NIFTYPET_MMR_AUXE_LIB})

# ------------------------------------------------------------- #
#                CREATE OUR LIBRARY AND INCLUDE OUR HEADERS
# ------------------------------------------------------------- #

set(dir gpu)
set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)
set(${dir_LIB_SOURCES}
  ProjectorByBinNiftyPETHelper.cxx
  ForwardProjectorByBinNiftyPET.cxx
  BackProjectorByBinNiftyPET.cxx
)

include(stir_lib_target)

target_link_libraries(gpu recon_buildblock modelling_buildblock display numerics_buildblock listmode_buildblock data_buildblock buildblock spatial_transformation_buildblock )


# ------------------------------------------------------------- #
#                ADD LIBRARIES AND HEADERS                      #
# ------------------------------------------------------------- #

# Add headers
TARGET_INCLUDE_DIRECTORIES(gpu PUBLIC ${CUDA_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(gpu PUBLIC ${NIFTYPET_INCLUDE_DIRS})

target_link_libraries(gpu ${CUDA_LIBRARIES})
target_link_libraries(gpu ${NIFTYPET_LIBRARIES})
