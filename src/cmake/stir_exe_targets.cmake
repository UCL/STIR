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

foreach(executable ${${dir_EXE_SOURCES}})
   add_executable(${executable} ${executable} ${STIR_REGISTRIES})
   target_link_libraries(${executable} ${STIR_LIBRARIES})
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Executables")
endforeach()

install(TARGETS ${${dir_EXE_SOURCES}} DESTINATION bin)
