# A macro to set the C++ version
# Based on https://stackoverflow.com/questions/10851247/how-to-activate-c-11-in-cmake

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

    if (CMAKE_VERSION VERSION_LESS "3.1")
      message(WARNING "Your CMake version is older than v3.1. Attempting to set C++ version to ${VERSION} with compiler flags but this might fail. Please upgrade your CMake.")
      if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_EXTENSIONS)
        set (CMAKE_CXX_FLAGS "-std=gnu++${VERSION} ${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
      else()
        set (CMAKE_CXX_FLAGS "-std=c++${VERSION} ${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
      endif ()
    else ()
      set (CMAKE_CXX_STANDARD ${VERSION} PARENT_SCOPE)
    endif ()

    message(STATUS "Using C++ version ${VERSION}")
endfunction(UseCXX)
