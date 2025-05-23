#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans
# Copyright 2016,2021 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file, to be included by CMakeLists.txt files that
# set dir_EXE_SOURCES to a list of executables that need to
# be compiled and installed

#
# NIKOS EFTHIMIOU
# Added dependency on ALL_HEADERS and ALL_INLINES in order
# to display .h and .inl files in Qt-Creator
foreach(source ${${dir_EXE_SOURCES}})
  if(BUILD_EXECUTABLES)
   get_filename_component(executable ${source} NAME_WE)
   add_executable(${executable} ${ALL_HEADERS} ${ALL_INLINES}  ${ALL_TXXS} ${source} ${STIR_REGISTRIES})
   target_link_libraries(${executable} ${STIR_LIBRARIES})
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Executables")
   target_include_directories(${executable} PUBLIC ${Boost_INCLUDE_DIR})
   install(TARGETS ${executable} DESTINATION bin)
  endif()
endforeach()
