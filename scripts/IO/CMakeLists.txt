#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details


set(dir scripts_IO)

set(dir_scripts ${dir}_SCRIPTS_IO)

set(${dir_scripts}
  convertSiemensInterfileToSTIR.sh 
  dumpSiemensDicomInfo.sh 
)

#include(stir_test_exe_targets)

# These don't need STIR executables, so always install
install(PROGRAMS ${${dir_scripts}} DESTINATION bin)
