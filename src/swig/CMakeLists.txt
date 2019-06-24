# Copyright 2012 Kris Thielemans
# Copyright 2014, 2018 University College London
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

if(BUILD_SWIG_PYTHON OR BUILD_SWIG_OCTAVE OR BUILD_SWIG_MATLAB)

  FIND_PACKAGE(SWIG 3.0 REQUIRED)
  INCLUDE("${SWIG_USE_FILE}")

  SET(CMAKE_SWIG_FLAGS -DSTART_NAMESPACE_STIR=\"namespace stir {\" -DEND_NAMESPACE_STIR=\"}\")

  SET_SOURCE_FILES_PROPERTIES(stir.i PROPERTIES CPLUSPLUS ON)
  # some include files in STIR have "#ifdef SWIG" statements, so we defined the variable here
  add_definitions(-D SWIG)

  if(BUILD_DOCUMENTATION)
    # We use doxy2swig to convert the doxygen-generated xml to a swig file
    # Note the "standard" CMake trick of first adding a custom command that generates a file, and then
    # a custom target that depends on the file.
    add_custom_command(
      OUTPUT STIR_DOXY2SWIG.i
      DEPENDS RUN_DOXYGEN
      COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/external_helpers/doxy2swig/doxy2swig.py -c
          ${CMAKE_BINARY_DIR}/xml/index.xml
          STIR_DOXY2SWIG.i
    )
    add_custom_target(
      RUN_DOXY2SWIG
      DEPENDS STIR_DOXY2SWIG.i)
    # Set a variable that will be passed to SWIG. It specifies the location of the generated .i file.
    # We need some interesting "escaping" to be able to pass a quoted string as a preprocessor variable to SWIG.
    # (The value of the preprocessor variables needs to have the quotes, as SWIG doesn't
    # substitute preprocessor variables inside quotes).
    # (SWIG_DOXY_OPTIONS will be empty if BUILD_DOCUMENTATION is off).
    set(SWIG_DOXY_OPTIONS -DDOXY2SWIG_XML_INCLUDE_FILE="\\\"${CMAKE_CURRENT_BINARY_DIR}/STIR_DOXY2SWIG.i\\\"")
    # Set a variable that will be used below to indicate dependency on the generated .i
    # (It will be empty if BUILD_DOCUMENTATION is off).
    set(SWIG_DOXY_TARGET RUN_DOXY2SWIG) # ${CMAKE_CURRENT_BINARY_DIR}/STIR_DOXY2SWIG.i)

    #configure_file(stir.i.in ${CMAKE_CURRENT_BINARY_DIR}/stir.i)
  endif()

endif()

if(BUILD_SWIG_PYTHON)

  # find Python interpreter. Needed for tests, and best to enforce consistency anyway.
  find_package(PythonInterp)
  # find libraries and include files
  # TODO would be better to use target_include_directories
  FIND_PACKAGE(PythonLibs REQUIRED)
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
  FIND_PACKAGE(Numpy REQUIRED)
  INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIRS})
  # TODO probably better to call the module stirpy or something
  # TODO -builtin option only appropriate for python
  # while the next statement sets it for all modules called stir
  SET(SWIG_MODULE_stir_EXTRA_FLAGS -builtin ${SWIG_DOXY_OPTIONS})
  if (CMAKE_VERSION VERSION_LESS "3.8")
    SWIG_ADD_MODULE(stir python stir.i ${STIR_REGISTRIES})
  else()
    SWIG_ADD_LIBRARY(stir LANGUAGE python TYPE MODULE SOURCES stir.i ${STIR_REGISTRIES})
  endif()
  SWIG_LINK_LIBRARIES(stir ${STIR_LIBRARIES} ${PYTHON_LIBRARIES})
  SET_TARGET_PROPERTIES(${SWIG_MODULE_stir_REAL_NAME} PROPERTIES
        LINK_FLAGS "${OpenMP_EXE_LINKER_FLAGS}")  
  if (SWIG_DOXY_TARGET)
    add_dependencies(${SWIG_MODULE_stir_REAL_NAME} ${SWIG_DOXY_TARGET})
  endif()
  CONFIGURE_FILE(./pyfragments.swg ./ COPYONLY)

  set(PYTHON_DEST ${CMAKE_INSTALL_PREFIX}/python CACHE PATH "Destination for python module")
  INSTALL(TARGETS ${SWIG_MODULE_stir_REAL_NAME} DESTINATION ${PYTHON_DEST})
  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/stir.py  DESTINATION ${PYTHON_DEST})
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

  SET(SWIG_MODULE_stiroct_EXTRA_FLAGS -module stiroct ${SWIG_DOXY_OPTIONS})
  if (CMAKE_VERSION VERSION_LESS "3.8")
    SWIG_ADD_MODULE(stiroct octave stir.i ${STIR_REGISTRIES})
  else()
    SWIG_ADD_LIBRARY(stiroct LANGUAGE octave TYPE MODULE SOURCES stir.i ${STIR_REGISTRIES})
  endif()
  SET_TARGET_PROPERTIES(${SWIG_MODULE_stiroct_REAL_NAME} PROPERTIES SUFFIX ${OCTAVE_SUFFIX} PREFIX "${OCTAVE_PREFIX}")
  SWIG_LINK_LIBRARIES(stiroct ${STIR_LIBRARIES} ${OCTAVE_LIBRARIES})
  if (SWIG_DOXY_TARGET)
    add_dependencies(${SWIG_MODULE_stiroct_REAL_NAME} ${SWIG_DOXY_TARGET})
  endif()

  # add OCTAVE_INCFLAGS to swig-generated file only, not to all files as 
  # 1) we don't need it at the moment 2) we'd need to change from -Ibla to bla
  #INCLUDE_DIRECTORIES(${OCTAVE_INCLUDE_PATH})
  SET_SOURCE_FILES_PROPERTIES( ${swig_generated_file_fullname}
        PROPERTIES COMPILE_FLAGS ${OCTAVE_INCFLAGS})

  set(OCTAVE_DEST ${CMAKE_INSTALL_PREFIX}/octave CACHE PATH "Destination for Octave module")
  INSTALL(TARGETS ${SWIG_MODULE_stiroct_REAL_NAME} DESTINATION ${OCTAVE_DEST})

endif (BUILD_SWIG_OCTAVE)

if (BUILD_SWIG_MATLAB)

  set(module_name stir)
  SET(SWIG_MODULE_stirMATLAB_EXTRA_FLAGS -module ${module_name} ${SWIG_DOXY_OPTIONS})
  # TODO depending on SWIG version add "-mexname stirMATLAB_wrap" above
  if (CMAKE_VERSION VERSION_LESS "3.8")
    SWIG_ADD_MODULE(stirMATLAB matlab stir.i ${STIR_REGISTRIES})
  else()
    SWIG_ADD_LIBRARY(stirMATLAB LANGUAGE matlab TYPE MODULE SOURCES stir.i ${STIR_REGISTRIES})
  endif()
  if (WIN32)
     set (Matlab_CXXLINKER_FLAGS "/EXPORT:mexFunction")
  endif()
  SET_TARGET_PROPERTIES(${SWIG_MODULE_stirMATLAB_REAL_NAME} PROPERTIES
        SUFFIX "_wrap.${MATLAB_MEX_EXT}" PREFIX "${MATLAB_PREFIX}"
        LINK_FLAGS "${Matlab_CXXLINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}"
        FOLDER "Matlab")  
  if (SWIG_DOXY_TARGET)
    add_dependencies(${SWIG_MODULE_stirMATLAB_REAL_NAME} ${SWIG_DOXY_TARGET})
  endif()
  SWIG_LINK_LIBRARIES(stirMATLAB ${STIR_LIBRARIES}  ${Matlab_LIBRARIES})

  include_directories(${Matlab_INCLUDE_DIRS})
  # disabled, as currently set via add_definitions in main CMakeLists.txt
  #SET_SOURCE_FILES_PROPERTIES( ${swig_generated_file_fullname}
  #      PROPERTIES COMPILE_FLAGS "${MATLAB_CXXFLAGS}")

  set(MATLAB_DEST ${CMAKE_INSTALL_PREFIX}/matlab CACHE PATH "Destination for Matlab module (relative to CMAKE_INSTALL_PREFIX)")
  INSTALL(TARGETS ${SWIG_MODULE_stirMATLAB_REAL_NAME} DESTINATION ${MATLAB_DEST})
  INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/+${module_name} DESTINATION ${MATLAB_DEST})
  file(GLOB SwigMatlabFiles "${CMAKE_CURRENT_BINARY_DIR}/Swig*.m")
  INSTALL(FILES ${SwigMatlabFiles} DESTINATION ${MATLAB_DEST})
  
endif (BUILD_SWIG_MATLAB)
