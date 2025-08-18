#
#
# Copyright 2014, University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir local_motion_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  rigid_object_transform_projdata.cxx
  rigid_object_transform_image.cxx
  non_rigid_transform.cxx
)

#include(stir_exe_targets)
foreach(executable ${${dir_EXE_SOURCES}})
   add_executable(${executable} ${executable} ${STIR_IO_REGISTRIES} )
   target_link_libraries(${executable} PUBLIC buildblock IO buildblock local_motion_buildblock buildblock IO buildblock listmode_buildblock display PRIVATE fmt)
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Executables")
endforeach()

install(TARGETS ${${dir_EXE_SOURCES}} DESTINATION bin)

