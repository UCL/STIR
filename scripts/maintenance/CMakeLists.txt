#
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory

set(dir maintenance_scripts)

set(dir_maintenance_scripts ${dir}_SCRIPTS)

set(${dir_maintenance_scripts}
)



#include(stir_test_exe_targets)

install(PROGRAMS ${${dir_maintenance_scripts}} DESTINATION bin)

