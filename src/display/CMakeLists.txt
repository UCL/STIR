#
#

set(dir display)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)
set(${dir}_LIB_SOURCES 
	  display_array
)

find_package(X11)
if(X11_FOUND)
  set(default_GRAPHICS X)
else()
  set(default_GRAPHICS PGM)
endif()
set(GRAPHICS ${default_GRAPHICS} 
    CACHE STRING "Type of graphics to use. Choose X, PGM or None")


#******* first find out which files we need and what compilation options

if( "${GRAPHICS}" STREQUAL "X")
	set(${dir}_LIB_SOURCES 
	  ${${dir}_LIB_SOURCES}
	  gen screengen screen
	)
	ADD_DEFINITIONS(-DSTIR_SIMPLE_BITMAPS -DSC_XWINDOWS)

elseif("${GRAPHICS}" STREQUAL "PGM")
	ADD_DEFINITIONS(-DSTIR_PGM)
elseif("$(GRAPHICS)" STREQUAL "MATHLINK")
  message(FATAL_ERROR "Mathlink (i.e. Mathematica) support needs to be fixed. sorry.")
# this presumably needs a Mathlink library, depends on your system though
#GRAPH_LIBS =
#GRAPHOPT = -DSTIR_MATHLINK
#$(dir)_LIB_SOURCES :=   mathlinkhelp.c
elseif("${GRAPHICS}" STREQUAL "None")
        # nothing to be done here
else()
        message(FATAL_ERROR "GRAPHICS has to be X, PGM or None")
endif()

include(stir_lib_target)

if( "${GRAPHICS}" STREQUAL "X")
	find_package(Curses REQUIRED)
	INCLUDE_DIRECTORIES(${X11_INCLUDE_DIR} ${CURSES_INCLUDE_DIR})
	target_link_libraries(${dir} ${X11_LIBRARIES} ${CURSES_LIBRARY})
endif()

target_link_libraries(${dir} buildblock)
