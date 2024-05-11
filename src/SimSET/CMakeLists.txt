#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir SimSET)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

# routines that need linking with the STIR library
set(${dir_EXE_SOURCES}
	conv_SimSET_projdata_to_STIR.cxx
	conv_to_SimSET_att_image.cxx
)

include(stir_exe_targets)

# stand-alone
if (BUILD_EXECUTABLES)
  add_executable(write_phg_image_info write_phg_image_info.c)
  SET_PROPERTY(TARGET write_phg_image_info PROPERTY FOLDER "Executables")
  install(TARGETS write_phg_image_info DESTINATION bin)
endif()
