
#========================================================================
# Author: Kris Thielemans
# Copyright (C) 2024 University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details
#=========================================================================

# cmake file for building the STIR demo. See the README.md and http://www.cmake.org.

cmake_minimum_required(VERSION 3.14.0)

PROJECT(myproject)

find_package(STIR 6.0 REQUIRED CONFIG)

add_executable(demo_create_image demo_create_image.cxx $<TARGET_OBJECTS:stir_registries>)
target_include_directories(demo_create_image PUBLIC "${STIR_INCLUDE_DIRS}")
target_link_libraries(demo_create_image PUBLIC "${STIR_LIBRARIES}")

install(TARGETS demo_create_image DESTINATION bin)
