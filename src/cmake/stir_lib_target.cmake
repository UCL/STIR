#
#
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

# cmake helper file, to be included by CMakeLists.txt files that
# set ${dir} to the target-name of the library and ${dir_LIB_SOURCES}
# to its sources

add_library(${dir} ${${dir_LIB_SOURCES}}    )
target_include_directories(${dir} PUBLIC 
  $<BUILD_INTERFACE:${STIR_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

SET_PROPERTY(TARGET ${dir} PROPERTY FOLDER "Libs")

install(TARGETS ${dir} EXPORT STIRTargets DESTINATION lib)
