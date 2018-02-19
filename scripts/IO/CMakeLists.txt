#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans
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
