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

#
# The following code has been modified in order to display .h and .inl files in Qt-Creator
#

#MESSAGE(STATUS "Configuring library ${dir}")
#set(G_HEADER_PATH "${PROJECT_SOURCE_DIR}/src/include/stir/")
#set(HEADER_PATH "${PROJECT_SOURCE_DIR}/src/include/stir/${dir}")

#MESSAGE(STATUS "Configuring library ${HEADER_PATH}")

#FILE(GLOB_RECURSE INC_ALL "${HEADER_PATH}/*.h")
#FILE(GLOB_RECURSE G_INC_ALL "${G_HEADER_PATH}/*.h")

#FILE(GLOB_RECURSE I_ INC_ALL "${HEADER_PATH}/*.inl")
#FILE(GLOB_RECURSE I_G_INC_ALL "${G_HEADER_PATH}/*.inl")

#MESSAGE(STATUS "headers: ${INC_ALL}")
#${INC_ALL} ${G_INC_ALL} ${I_INC_ALL} ${I_G_INC_ALL}

add_library(${dir}  ${${dir_LIB_SOURCES}}  )
SET_PROPERTY(TARGET ${dir} PROPERTY FOLDER "Libs")

install(TARGETS ${dir} DESTINATION lib)
