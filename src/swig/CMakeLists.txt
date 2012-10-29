#
# $Id$
#
# Copyright 2012-06-05 - $Date$ Kris Thielemans

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

# cmake file for building interfaces to STIR usgin SWIG. See the STIR User's Guide and http://www.cmake.org.

set(dir swig)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

# TODO -builtin option only appropriate for python
SET(CMAKE_SWIG_FLAGS -builtin  -DSTART_NAMESPACE_STIR=\"namespace stir {\" -DEND_NAMESPACE_STIR=\"}\")

SET_SOURCE_FILES_PROPERTIES(stir.i PROPERTIES CPLUSPLUS ON)

add_definitions(-D SWIG)

SWIG_ADD_MODULE(stir python stir.i ${STIR_REGISTRIES})
SWIG_LINK_LIBRARIES(stir ${STIR_LIBRARIES} ${PYTHON_LIBRARIES})

set(PYTHON_DEST ${CMAKE_INSTALL_PREFIX}/python CACHE PATH "Destination for python module")
INSTALL(TARGETS _stir DESTINATION ${PYTHON_DEST})
INSTALL(FILES ${CMAKE_BINARY_DIR}/${dir}/stir.py  DESTINATION ${PYTHON_DEST})
