#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file for building STIR.

set(dir iterative_POSMAPOSL)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
   POSMAPOSL.cxx
)

include(stir_exe_targets)
