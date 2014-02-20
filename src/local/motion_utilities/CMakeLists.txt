#
#
# Copyright 2014, University College London

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

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir local_motion_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  rigid_object_transform_projdata
  rigid_object_transform_image
)

#include(${CMAKE_SOURCE_DIR}/stir_exe_targets.cmake)
foreach(executable ${${dir_EXE_SOURCES}})
   add_executable(${executable} ${executable} ${CMAKE_HOME_DIRECTORY}/IO/IO_registries.cxx )
   target_link_libraries(${executable} buildblock IO buildblock local_motion_buildblock buildblock IO buildblock listmode_buildblock)
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Executables")
endforeach()

install(TARGETS ${${dir_EXE_SOURCES}} DESTINATION bin)

