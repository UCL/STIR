#
#
# Copyright 2011-07-01 - 2012 Kris Thielemans
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file handling all files in this subdirectory

set(dir SimSET_scripts)

set(dir_scripts ${dir}_SCRIPTS)

set(${dir_scripts}
add_SimSET_results.sh
conv_SimSET_projdata_to_STIR.sh
make_hv_from_Simset_params.sh
mult_num_photons.sh
run_SimSET.sh
SimSET_STIR_names.sh
stir_image_to_simset_object.sh
)


#include(stir_test_exe_targets)

if (BUILD_EXECUTABLES)
  install(PROGRAMS ${${dir_scripts}} DESTINATION bin)
endif()
