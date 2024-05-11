#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir scatter_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	estimate_scatter.cxx
	create_tail_mask_from_ACFs.cxx
	upsample_and_fit_single_scatter.cxx
	simulate_scatter.cxx
)

include(stir_exe_targets)
