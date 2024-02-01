
#========================================================================
# Author: Kris Thielemans
# Copyright 2016 - 2020 University College London
# Copyright 2016 - 2020 Science Technology Facilities Council
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

# cmake file for building SIRF. See the SIRF User's Guide and http://www.cmake.org.

cmake_minimum_required(VERSION 3.14.0)

PROJECT(myproject)

find_package(STIR 6.0 REQUIRED CONFIG)

add_executable(demo_create_image demo_create_image.cxx)
target_include_directories(demo_create_image PUBLIC "${STIR_INCLUDE_DIRS}")
target_link_libraries(demo_create_image PUBLIC "${STIR_LIBRARIES}")

install(TARGETS demo_create_image DESTINATION bin)
