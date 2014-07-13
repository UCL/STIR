# Copyright 2012 Kris Thielemans
# Copyright 2014 University College London
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

# cmake file for building interfaces to STIR using SWIG. See the STIR User's Guide and http://www.cmake.org.

set(dir swig)

option(BUILD_SWIG_PYTHON
       "Use SWIG to generate python bindings for STIR (experimental)" OFF)

option(BUILD_SWIG_OCTAVE
       "Use SWIG to generate octave bindings for STIR (experimental)" OFF)

option(BUILD_SWIG_MATLAB
       "Use SWIG to generate matlab bindings for STIR (experimental)" OFF)

# Octave support doesn't really work very well yet, so hide it a bit
mark_as_advanced(BUILD_SWIG_OCTAVE)

if(BUILD_SWIG_PYTHON OR BUILD_SWIG_OCTAVE OR BUILD_SWIG_MATLAB)

  FIND_PACKAGE(SWIG REQUIRED)
  INCLUDE("${SWIG_USE_FILE}")

  SET(CMAKE_SWIG_FLAGS -DSTART_NAMESPACE_STIR=\"namespace stir {\" -DEND_NAMESPACE_STIR=\"}\")

  SET_SOURCE_FILES_PROPERTIES(stir.i PROPERTIES CPLUSPLUS ON)
  # some include files in STIR have "#ifdef SWIG" statements, so we defined the variable here
  add_definitions(-D SWIG)
endif()

if(BUILD_SWIG_PYTHON)

  FIND_PACKAGE(PythonLibs)
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
  # TODO probably better to call the module stirpy or something
  # TODO -builtin option only appropriate for python
  # while the next statement sets it for all modules called stir
  SET(SWIG_MODULE_stir_EXTRA_FLAGS -builtin)
  SWIG_ADD_MODULE(stir python stir.i ${STIR_REGISTRIES})
  SWIG_LINK_LIBRARIES(stir ${STIR_LIBRARIES} ${PYTHON_LIBRARIES})

  set(PYTHON_DEST ${CMAKE_INSTALL_PREFIX}/python CACHE PATH "Destination for python module")
  INSTALL(TARGETS ${SWIG_MODULE_stir_REAL_NAME} DESTINATION ${PYTHON_DEST})
  INSTALL(FILES ${CMAKE_BINARY_DIR}/${dir}/stir.py  DESTINATION ${PYTHON_DEST})
  INSTALL(FILES stirextra.py  DESTINATION ${PYTHON_DEST})

endif(BUILD_SWIG_PYTHON)

if (BUILD_SWIG_OCTAVE)
  # we will use mkoctfile to get configuration info, but not use it for the actual build.
  # it'd be hard to get cmake to work with that (specifying all STIR flags and libraries for instance)
  # also, mkoctfile currently doesn't take .so files

  FIND_PROGRAM(MKOCTFILE mkoctfile)
  IF(NOT MKOCTFILE)
    MESSAGE(FATAL_ERROR "mkoctfile was not found. We need it to build the STIR interface for Octave.")
  ENDIF(NOT MKOCTFILE)

  # TODO need to test if SHARED is on

  # this doesn't exist in standard cmake and we don't need it I guess
  #FIND_PACKAGE(Octave)

  # get necessary include/link flags and libraries

  EXECUTE_PROCESS(COMMAND ${MKOCTFILE} -p INCFLAGS       OUTPUT_VARIABLE OCTAVE_INCFLAGS)
  EXECUTE_PROCESS(COMMAND ${MKOCTFILE} -p OCTAVE_LFLAGS  OUTPUT_VARIABLE OCTAVE_LINKFLAGS)
  EXECUTE_PROCESS(COMMAND ${MKOCTFILE} -p OCTAVE_LIBS    OUTPUT_VARIABLE OCTAVE_LIBS)
  # get rid of newlines
  string(REGEX REPLACE "[\r\n]" ""  OCTAVE_INCFLAGS ${OCTAVE_INCFLAGS})

  #    SET(OCTAVE_FOUND TRUE)
  SET(OCTAVE_LIBS ${OCT_LINKFLAGS} ${OCTAVE_LIBS})
  SET(OCTAVE_SUFFIX ".oct")
  SET(OCTAVE_PREFIX "")

  SET(SWIG_MODULE_stiroct_EXTRA_FLAGS -module stiroct)
  SWIG_ADD_MODULE(stiroct octave stir.i ${STIR_REGISTRIES})
  SET_TARGET_PROPERTIES(${SWIG_MODULE_stiroct_REAL_NAME} PROPERTIES SUFFIX ${OCTAVE_SUFFIX} PREFIX "${OCTAVE_PREFIX}")
  SWIG_LINK_LIBRARIES(stiroct ${STIR_LIBRARIES} ${OCTAVE_LIBRARIES})

  # add OCTAVE_INCFLAGS to swig-generated file only, not to all files as 
  # 1) we don't need it at the moment 2) we'd need to change from -Ibla to bla
  #INCLUDE_DIRECTORIES(${OCTAVE_INCLUDE_PATH})
  SET_SOURCE_FILES_PROPERTIES( ${swig_generated_file_fullname}
        PROPERTIES COMPILE_FLAGS ${OCTAVE_INCFLAGS})

  set(OCTAVE_DEST ${CMAKE_INSTALL_PREFIX}/octave CACHE PATH "Destination for Octave module")
  INSTALL(TARGETS ${SWIG_MODULE_stiroct_REAL_NAME} DESTINATION ${OCTAVE_DEST})

endif (BUILD_SWIG_OCTAVE)

if (BUILD_SWIG_MATLAB)
  FIND_PACKAGE(MATLAB)
  include(BuildMex)

  set(module_name stir)
  SET(SWIG_MODULE_stirMATLAB_EXTRA_FLAGS -module ${module_name})
  SWIG_ADD_MODULE(stirMATLAB matlab stir.i ${STIR_REGISTRIES})
  SET_TARGET_PROPERTIES(${SWIG_MODULE_stirMATLAB_REAL_NAME} PROPERTIES
        SUFFIX "_wrap.${MATLAB_MEX_EXT}" PREFIX "${MATLAB_PREFIX}"
        LINK_FLAGS "${mexLdFlags}"
        FOLDER "Matlab")  
  SWIG_LINK_LIBRARIES(stirMATLAB ${STIR_LIBRARIES} ${mexCxxLibs})

  SET_SOURCE_FILES_PROPERTIES( ${swig_generated_file_fullname}
        PROPERTIES COMPILE_FLAGS "${mexCxxFlags}")
  # TODO we really need this for all files, at least on Linux 
  # add_definitions(${mexCxxFlags})
  # Reason: matlab needs mex files to be compiled with -fPIC, but that means all 
  # linked libraries need to have been compiled with -fPIC as well.
  # However, that means we need to move this add_definitions stuff into src/CMakeLists.txt

  set(MATLAB_DEST ${CMAKE_INSTALL_PREFIX}/matlab CACHE PATH "Destination for Matlab module (relative to CMAKE_INSTALL_PREFIX)")
  INSTALL(TARGETS ${SWIG_MODULE_stirMATLAB_REAL_NAME} DESTINATION ${MATLAB_DEST})
  INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/+${module_name} DESTINATION ${MATLAB_DEST})
  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/SwigRef.m DESTINATION ${MATLAB_DEST})
  
endif (BUILD_SWIG_MATLAB)
