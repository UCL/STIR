#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory

set(dir scripts_IO)

set(dir_scripts ${dir}_SCRIPTS_IO)

set(${dir_scripts}
  convertSiemensInterfileToSTIR.sh 
  dumpSiemensDicomInfo.sh 
)

#include(stir_test_exe_targets)

# only install if executables are installed as well, as the scripts need them
if (BUILD_EXECUTABLES)
  install(PROGRAMS ${${dir_scripts}} DESTINATION bin)
endif()
