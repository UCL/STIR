#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2011 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir listmode_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  lm_to_projdata.cxx
  lm_to_projdata_bootstrap.cxx
  lm_to_projdata_with_random_rejection.cxx
  lm_fansums.cxx
  list_lm_info.cxx
  list_lm_events.cxx
  list_lm_countrates.cxx
  )

if (HAVE_ECAT)
  # yes, the LLN files seem to be there, so we can compile more
  list(APPEND ${dir_EXE_SOURCES}
	scan_sgl_file.cxx
	print_sgl_values.cxx
	rebin_sgl_file.cxx
	add_ecat7_header_to_sgl.cxx
  )
endif()

if (STIR_WITH_NiftyPET_PROJECTOR)
  list(APPEND ${dir_EXE_SOURCES} conv_NiftyPET_stir)
  list(APPEND ${dir_EXE_SOURCES} lm_to_projdata_NiftyPET)
endif()


include(stir_exe_targets)
