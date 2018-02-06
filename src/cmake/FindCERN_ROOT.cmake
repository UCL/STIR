#
# Simple (or not) script to find CERN ROOT include dir and libs.
# @Author Nikos Efthimiou (nikos.efthimiou AT gmail.com)
# If CERN_ROOT_LIBRARIES_DIRS and CERN_ROOT_INCLUDE_DIRS are set
# then it will try to include headers from that location and link to
# the basic libraries that we know that we need.
# If the variables have not been set then it will try to locate a global
# installation and it will link to all available libraries.
# We give priority to local ROOT installations, because the user might
# prefer to use a specific version, rather than the one installed in the
# system.

# This file contains lines from FindROOT.cmake distributed in ROOT 6.08.
# Therefore, this file is presumably licensed under the LGPL 2.1.

if (CERN_ROOT_LIBRARIES_DIRS AND CERN_ROOT_INCLUDE_DIRS)
    find_path(CERN_ROOT_INCLUDE_DIRS NAME TROOT.h
        DOC "location of ROOT include files")
    message(STATUS "CERN_ROOT_INCLUDE_DIRS:" ${CERN_ROOT_INCLUDE_DIRS})

    find_library(CERN_ROOT_LIBRARIES_DIRS NAME Core
        DOC "location of ROOT library")

    set(CERN_ROOT_LIBRARIES "-L${CERN_ROOT_LIBRARIES_DIRS} -lCore -lCint -lRIO -lNet -lTree")
else()
    find_program(CERN_ROOT_CONFIG "root-config" )

    if (CERN_ROOT_CONFIG)

        set (root_incl_arg "--incdir")

        execute_process(COMMAND ${CERN_ROOT_CONFIG} ${root_incl_arg} OUTPUT_VARIABLE
            CERN_ROOT_INCLUDE_DIRS)

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

	execute_process(
	    COMMAND ${CERN_ROOT_CONFIG} --version
	        OUTPUT_VARIABLE CERN_ROOT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)

    endif(CERN_ROOT_CONFIG)
endif()

if (CERN_ROOT_LIBRARIES)
  message(STATUS "AVAILABLE ROOT LIBRARIES:" ${CERN_ROOT_LIBRARIES})
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CERN_ROOT "CERN ROOT not found. If you do have it, add root-config to your path" CERN_ROOT_LIBRARIES CERN_ROOT_INCLUDE_DIRS)
