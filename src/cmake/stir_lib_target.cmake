#
#
# Copyright 2011-07-01 - 2013 Kris Thielemans
# Copyright 2016, 2019, 2020 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file, to be included by CMakeLists.txt files that
# set ${dir} to the target-name of the library and ${dir_LIB_SOURCES}
# to its sources

add_library(${dir} ${${dir_LIB_SOURCES}}    )
target_include_directories(${dir} PUBLIC 
  $<BUILD_INTERFACE:${STIR_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

# make sure that if you use STIR, the compiler will be set to at least C++11
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.8.0")
  target_compile_features(${dir} PUBLIC cxx_std_11)
else()
  # Older CMake didn't have cxx_std_11 yet, but using auto will presumably force it anyway
  target_compile_features(${dir} PUBLIC cxx_auto_type)
endif()
target_include_directories(${dir} PUBLIC ${Boost_INCLUDE_DIR})

SET_PROPERTY(TARGET ${dir} PROPERTY FOLDER "Libs")

install(TARGETS ${dir} EXPORT STIRTargets DESTINATION lib)
