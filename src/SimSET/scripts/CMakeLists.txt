#
#
# Copyright 2011-07-01 - 2012 Kris Thielemans
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


#include(../stir_test_exe_targets.cmake)

install(PROGRAMS ${${dir_scripts}} DESTINATION bin)

