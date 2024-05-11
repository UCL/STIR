# Copyright 2016, 2018 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file, to be included by the root CMakeLists.txt
# it sets 2 variables ALL_HEADERS ALL_INLINES
# These are used for doxygen dependencies etc

# The following macro finds all STIR headers and inlines.

# The original MACRO was an adaptation from
# https://cmake.org/pipermail/cmake/2012-June/050674.html
# but we don't need the complicated stuff

MACRO(find_all_header_files return_h_list return_i_list return_txx_list HEADER_DIR)
    #FILE(GLOB_RECURSE new_list "${HEADER_DIR}")
    FILE(GLOB_RECURSE h_list "${HEADER_DIR}/*.h")
    FILE(GLOB_RECURSE i_list "${HEADER_DIR}/*.inl")
    FILE(GLOB_RECURSE txx_list "${HEADER_DIR}/*.txx")

    #SET(dir_list "")
    #FOREACH(file_path ${new_list})
    #    GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
    #    SET(dir_list ${dir_list} ${dir_path})
    #    FILE(GLOB_RECURSE h_list "${dir_path}/*.h")
    #    FILE(GLOB_RECURSE i_list "${dir_path}/*.inl")
    #ENDFOREACH()

    #LIST(REMOVE_DUPLICATES h_list)
    #LIST(REMOVE_DUPLICATES i_list)

    LIST(APPEND ${return_h_list} ${h_list})
    LIST(APPEND ${return_i_list} ${i_list})
    LIST(APPEND ${return_txx_list} ${txx_list})
ENDMACRO()

