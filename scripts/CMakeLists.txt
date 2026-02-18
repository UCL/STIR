#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans
# Copyright 2022 University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

set(dir scripts)

#set(dir_scripts ${dir}_SCRIPTS)

set(dir_scripts1
  stir_subtract
  stir_divide
  stir_print_voxel_sizes.sh
  estimate_scatter.sh
  zoom_att_image.sh
  get_num_voxels.sh
  create_fdef_from_listmode.sh
  )
# scripts that don't use stir utilities and might be useful from python etc
set(dir_scripts2
  count 
  force_view_offset_to_zero.sh
  )


#include(stir_test_exe_targets)

# only install if executables are installed as well, as the scripts need them
if (BUILD_EXECUTABLES)
  install(PROGRAMS ${dir_scripts1} DESTINATION bin)
endif()

install(PROGRAMS ${dir_scripts2} DESTINATION bin)

add_subdirectory(IO)
