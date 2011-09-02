#
# $Id$
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - $Date$ Kris Thielemans

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
# set dir_SIMPLE_TEST_EXE_SOURCES and dir_INVOLVED_TEST_EXE_SOURCES
# to a list of executables that need to be compiled.
# Moreover, we will use ADD_TEST to create a test for
# the files in dir_SIMPLE_TEST_EXE_SOURCES (assuming these
# don't need any parameters).

foreach(executable ${${dir_SIMPLE_TEST_EXE_SOURCES}})
   add_executable(${executable} ${executable}.cxx ${STIR_REGISTRIES})
   target_link_libraries(${executable} ${STIR_LIBRARIES})
   ADD_TEST(${executable} ${CMAKE_CURRENT_BINARY_DIR}/${executable})
endforeach()

foreach(executable ${${dir_INVOLVED_TEST_EXE_SOURCES}})
   add_executable(${executable} ${executable}.cxx ${STIR_REGISTRIES})
   target_link_libraries(${executable} ${STIR_LIBRARIES})
endforeach()

