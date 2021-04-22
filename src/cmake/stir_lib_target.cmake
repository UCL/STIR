#
#
# Copyright 2011-07-01 - 2013 Kris Thielemans

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

target_include_directories(${dir} PUBLIC ${Boost_INCLUDE_DIR})

SET_PROPERTY(TARGET ${dir} PROPERTY FOLDER "Libs")

install(TARGETS ${dir} EXPORT STIRTargets DESTINATION lib)
