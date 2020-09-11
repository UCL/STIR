#
# Simple (or not) script to find CERN ROOT include dir and libs.
# @Author Nikos Efthimiou (nikos.efthimiou AT gmail.com)
# @Author Kris Thielemans

# Attempts to find_package(ROOT), If that fails, use root-config.
# If that fails, try and find TROOT.h and libCore*
# Uses the ROOT_DIR CMake variable, then ROOTSYS, and then ROOTSYS environment variable
#
# Defines CERN_ROOT_LIBRARIES, CERN_ROOT_INCLUDE_DIRS and CERN_ROOT_VERSION
#
# when find_package(ROOT) worked, it will also set
# ROOT_INCLUDE_DIRS - include directories for ROOT
# ROOT_DEFINITIONS  - compile definitions needed to use ROOT
# ROOT_LIBRARIES    - libraries to link against
# ROOT_USE_FILE     - path to a CMake module which may be included to help

# Set CERN_ROOT_DEBUG for some extra info

# This file contains lines from FindROOT.cmake distributed in ROOT 6.08.
# Therefore, this file is presumably licensed under the LGPL 2.1.
# New parts Copyright 2016, 2020 University College London

if (NOT DEFINED ROOTSYS)
  set(ROOTSYS "$ENV{ROOTSYS}")
endif()

if (NOT DEFINED ROOT_DIR AND DEFINED ROOTSYS)
  set(ROOT_DIR:PATH ${ROOTSYS}/cmake)
endif()

find_package(ROOT ${CERN_ROOT_FIND_VERSION} QUIET)
if (ROOT_FOUND)
  if (CERN_ROOT_DEBUG)
    message(STATUS "Found ROOTConfig.cmake, so translating to old CERN_ROOT variable names")
  endif()
  set(CERN_ROOT_VERSION ${ROOT_VERSION})
  set(CERN_ROOT_INCLUDE_DIRS ${ROOT_INCLUDE_DIRS})
  set(CERN_ROOT_LIBRARIES ${ROOT_LIBRARIES})

else()

  ### Old work-arounds. Should be removed later really

    if (CERN_ROOT_DEBUG)
      message(STATUS "Did not find ROOTConfig.cmake, so trying via root-config")
    endif()
    find_program(CERN_ROOT_CONFIG "root-config" HINTS "${ROOTSYS}" )

    if (CERN_ROOT_CONFIG)

        if (CERN_ROOT_DEBUG)
          message(STATUS "Finding ROOT location etc via ${CERN_ROOT_CONFIG}")
        endif()

        execute_process(COMMAND ${CERN_ROOT_CONFIG} --incdir OUTPUT_VARIABLE
            CERN_ROOT_INCLUDE_DIRS
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        # Attempt fo find libraries from root-config. However, this doesn't work if
        # not all libraries are installed (as root-config lists them anyway).
        # set (root_lib_arg "--libs")
        # execute_process(COMMAND ${CERN_ROOT_CONFIG} ${root_lib_arg} OUTPUT_VARIABLE
        #    TCERN_ROOT_LIBRARIES)
        # string (STRIP "${TCERN_ROOT_LIBRARIES}" CERN_ROOT_LIBRARIES)

        # Do an explicit search
        # Lines copied from FindROOT.cmake distributed with ROOT v6.08/06
        execute_process(
            COMMAND ${CERN_ROOT_CONFIG} --libdir
            OUTPUT_VARIABLE CERN_ROOT_LIBRARY_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

	execute_process(
	    COMMAND ${CERN_ROOT_CONFIG} --version
	        OUTPUT_VARIABLE CERN_ROOT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)

    else()

        # no root-config
        if (CERN_ROOT_DEBUG)
          message(STATUS "Did not find root-config, so trying via TRoot.h and the Core library")
        endif()
        find_path(CERN_ROOT_INCLUDE_DIR TROOT.h HINTS "${ROOTSYS}"
            DOC "location of ROOT include files")
        set(CERN_ROOT_INCLUDE_DIRS "${CERN_ROOT_INCLUDE_DIR}")
        
        find_library(CERN_ROOT_Core_LIBRARY Core HINTS "${ROOTSYS}" "${CERN_ROOT_INCLUDE_DIRS}/.."
            DOC "location of ROOT libraries")
        
        if (CERN_ROOT_Core_LIBRARY)
            get_filename_component(CERN_ROOT_LIBRARIES_DIR "${CERN_ROOT_Core_LIBRARY}" DIRECTORY CACHE)
        endif()
        
        set(version_file ${CERN_ROOT_INCLUDE_DIRS}/RVersion.h)
        if (EXISTS ${version_file})
            if (CERN_ROOT_DEBUG)
               message(STATUS "Attempting to get ROOT version from ${version_file}")
            endif()
            file(STRINGS ${version_file} version_line REGEX "define ROOT_RELEASE ")
            if (${version_line} MATCHES ".*ROOT_RELEASE \"\(.+\)\"")
                set(CERN_ROOT_VERSION "${CMAKE_MATCH_1}")
            endif()
        else()
            if (CERN_ROOT_INCLUDE_DIRS)
              message(WARNING "Could not find ${version_file}")
            endif()
        endif()
    endif()

    set(CERN_ROOT_LIBRARY_DIRS ${CERN_ROOT_LIBRARY_DIR})
    
    set(_rootlibs Core RIO Net Hist Graf Graf3d Gpad Tree Rint Postscript Matrix Physics MathCore Thread MultiProc)
    set(CERN_ROOT_LIBRARIES)
    foreach(_cpt ${_rootlibs} ${CERN_ROOT_FIND_COMPONENTS})
      find_library(CERN_ROOT_${_cpt}_LIBRARY ${_cpt} HINTS ${CERN_ROOT_LIBRARY_DIR})
      if(CERN_ROOT_${_cpt}_LIBRARY)
        mark_as_advanced(CERN_ROOT_${_cpt}_LIBRARY)
        list(APPEND CERN_ROOT_LIBRARIES ${CERN_ROOT_${_cpt}_LIBRARY})
        if(CERN_ROOT_FIND_COMPONENTS)
          list(REMOVE_ITEM CERN_ROOT_FIND_COMPONENTS ${_cpt})
        endif()
      endif()
    endforeach()
    if(CERN_ROOT_LIBRARIES)
      list(REMOVE_DUPLICATES CERN_ROOT_LIBRARIES)
    endif()

endif()

if (CERN_ROOT_DEBUG)
  message(STATUS "CERN_ROOT_INCLUDE_DIRS: ${CERN_ROOT_INCLUDE_DIRS}")
  message(STATUS "AVAILABLE ROOT LIBRARIES: ${CERN_ROOT_LIBRARIES}")
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CERN_ROOT "CERN ROOT not found. If you do have it, set ROOT_DIR (preferred), ROOTSYS or add root-config to your path" CERN_ROOT_VERSION CERN_ROOT_LIBRARIES CERN_ROOT_INCLUDE_DIRS)
