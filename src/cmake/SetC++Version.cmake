# A macro to set the C++ version
# Based on https://stackoverflow.com/questions/10851247/how-to-activate-c-11-in-cmake
macro(UseCXX VERSION)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # Strictly speaking, this enables C++11 (or whatever) with GNU extensions.
      # However, this is what recent CMake does with CMAKE_CXX_STANDARD,
      # so we'll do the same.
      set (CMAKE_CXX_FLAGS "-std=gnu++${VERSION} ${CMAKE_CXX_FLAGS}")
    else()
      set (CMAKE_CXX_FLAGS "-std=c++${VERSION} ${CMAKE_CXX_FLAGS}")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD ${VERSION})
  endif ()
endmacro(UseCXX)
