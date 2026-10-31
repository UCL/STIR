#
#
# Copyright (C) 2026 University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

set(dir iterative_PreconditionedGradient)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	PreconditionedGradientReconstruction.cxx
)

include(stir_lib_target)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	PreconditionedGradient.cxx
)

include(stir_exe_targets)

if (STIR_MPI)
  SET_PROPERTY(TARGET PreconditionedGradient PROPERTY LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endif()

target_link_libraries(iterative_PreconditionedGradient PUBLIC ${STIR_RECON_BUILDBLOCK_LIB})
