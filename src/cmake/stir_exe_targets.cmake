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

# cmake helper file, to be included by CMakeLists.txt files that
# set dir_EXE_SOURCES to a list of executables that need to
# be compiled and installed

#
# NIKOS EFTHIMIOU
# The following code has been modified in order to display .h and .inl files in Qt-Creator
# The MACRO is an adaptation from https://cmake.org/pipermail/cmake/2012-June/050674.html
#

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/src/include/stir")

MACRO(HEADER_DIRECTORIES return_h_list return_i_list)
    FILE(GLOB_RECURSE new_list "${HEADER_DIR}")
    FILE(GLOB_RECURSE h_list "${HEADER_DIR}/*.h")
    FILE(GLOB_RECURSE i_list "${HEADER_DIR}/*.inl")

    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
        FILE(GLOB_RECURSE h_list "${dir_path}/*.h")
        FILE(GLOB_RECURSE i_list "${dir_path}/*.inl")
    ENDFOREACH()

    LIST(REMOVE_DUPLICATES h_list)
    LIST(REMOVE_DUPLICATES i_list)

    SET(${return_h_list} ${h_list})
    SET(${return_i_list} ${i_list})
ENDMACRO()

HEADER_DIRECTORIES(ALL_HEADERS ALL_INLINES)

foreach(executable ${${dir_EXE_SOURCES}})
   add_executable(${executable} ${ALL_HEADERS} ${ALL_INLINES} ${executable} ${STIR_REGISTRIES})
   target_link_libraries(${executable} ${STIR_LIBRARIES})
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Executables")
endforeach()

install(TARGETS ${${dir_EXE_SOURCES}} DESTINATION bin)
