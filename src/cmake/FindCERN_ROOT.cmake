#
# Simple (or not) script to find CERN ROOT include dir and libs.
# @Author Nikos Efthimiou (nikos.efthimiou AT gmail.com)
# If CERN_ROOT_LIBRARIES_DIRS and CERN_ROOT_INCLUDE_DIRS are set
# then it will try to include headers from that location and link to
# the basic libraries that we know that we need.
# If the variables have not been set then it will try to locate a global
# installation and it will link to all available libraries.
# We give priority to local ROOT installations, because the user might
# prefer to use a specific version, rathen than the one installed in the
# system.

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
        set (root_lib_arg "--libs")

        execute_process(COMMAND ${CERN_ROOT_CONFIG} ${root_incl_arg} OUTPUT_VARIABLE
            CERN_ROOT_INCLUDE_DIRS)

        execute_process(COMMAND ${CERN_ROOT_CONFIG} ${root_lib_arg} OUTPUT_VARIABLE
            TCERN_ROOT_LIBRARIES)

        string (STRIP "${TCERN_ROOT_LIBRARIES}" CERN_ROOT_LIBRARIES)

        #string (REPLACE " " ";" CERN_ROOT_LIBRARIES ${LCERN_ROOT_LIBRARIES})

	execute_process(
	    COMMAND ${CERN_ROOT_CONFIG} --version
	        OUTPUT_VARIABLE CERN_ROOT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)

    endif(CERN_ROOT_CONFIG)
endif()

message(STATUS "AVAILABLE ROOT LIBRARIES:" ${CERN_ROOT_LIBRARIES})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CERN_ROOT "CERN ROOT not found. If you do have it, set the missing variables" CERN_ROOT_LIBRARIES CERN_ROOT_INCLUDE_DIRS)
