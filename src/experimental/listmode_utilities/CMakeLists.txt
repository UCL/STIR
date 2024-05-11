#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2012 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir local_listmode_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

if (HAVE_ECAT)
  # yes, the LLN files seem to be there, so we can compile more
  # list(APPEND ${dir_EXE_SOURCES}
  # )
endif()



include(stir_exe_targets)
