#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory

set(dir scripts)

set(dir_scripts ${dir}_SCRIPTS)

set(${dir_scripts}
  stir_subtract 
  stir_divide 
  count 
  stir_print_voxel_sizes.sh
  estimate_scatter.sh
  zoom_att_image.sh
  get_num_voxels.sh
  create_fdef_from_listmode.sh
  force_view_offset_to_zero.sh
  )



#include(stir_test_exe_targets)

# only install if executables are installed as well, as the scripts need them
if (BUILD_EXECUTABLES)
  install(PROGRAMS ${${dir_scripts}} DESTINATION bin)
endif()

add_subdirectory(IO)
