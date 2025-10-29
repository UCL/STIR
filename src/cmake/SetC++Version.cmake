# A macro to set the C++ version

# Copyright (C) 2017, 2020 University College London
# Author Kris Thielemans

# sets minimum C++ version (sorry for the name of the macro)
function(UseCXX VERSION)
    if (NOT DEFINED VERSION)
       message(FATAL_ERROR "UseCXX expects a version argument")
    endif()

    if (DEFINED CMAKE_CXX_STANDARD)
        if (${CMAKE_CXX_STANDARD} GREATER ${VERSION})
           if (${CMAKE_CXX_STANDARD} STREQUAL 98 OR ${VERSION} STREQUAL 98)
               message(WARNING "UseCXX function does not cope well with version 98. Using ${VERSION}")
           else()
	       message(STATUS "CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD} was already more recent than version ${VERSION} asked. Keeping the former.")
               set(VERSION ${CMAKE_CXX_STANDARD})
           endif()
        endif()
    endif()

    set (CMAKE_CXX_STANDARD ${VERSION} PARENT_SCOPE)
    set (CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)

    message(STATUS "Using C++ version ${VERSION}")
endfunction(UseCXX)
