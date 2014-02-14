#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

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

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir SimSET)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

# routines that need linking with the STIR library
set(${dir_EXE_SOURCES}
	conv_SimSET_projdata_to_STIR
	conv_to_SimSET_att_image
)

include(../stir_exe_targets.cmake)

# stand-alone
add_executable(write_phg_image_info write_phg_image_info.c)
SET_PROPERTY(TARGET write_phg_image_info PROPERTY FOLDER "Executables")
install(TARGETS write_phg_image_info DESTINATION bin)
