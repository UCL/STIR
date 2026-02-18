#
#

set(dir display)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)
set(${dir}_LIB_SOURCES 
	  display_array.cxx
)

find_package(X11)
if(X11_FOUND)
  set(default_GRAPHICS X)
else()
  set(default_GRAPHICS PGM)
endif()
set(GRAPHICS ${default_GRAPHICS} 
    CACHE STRING "Type of graphics to use. Choose X, PGM or None")

set(TARGET ${STIR_BUILDBLOCK_LIB})

#******* first find out which files we need and what compilation options

if( "${GRAPHICS}" STREQUAL "X")
	set(${dir}_LIB_SOURCES 
	  ${${dir}_LIB_SOURCES}
	  gen.c screengen.c screen.c
	)
	set(DISPLAY_COMPILE_DEFINITIONS "STIR_SIMPLE_BITMAPS;SC_XWINDOWS")

elseif("${GRAPHICS}" STREQUAL "PGM")
	set(DISPLAY_COMPILE_DEFINITIONS STIR_PGM)
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

# add preprocessor defines to the compilation flags
# The following could (should?) have worked, but it doesn't. The definitions are not added.
#foreach(SOURCE ${${dir_LIB_SOURCES}})
#  set_source_files_properties(${SOURCE} PROPERTIES COMPILE_DEFINITIONS "${DISPLAY_COMPILE_DEFINITIONS}")
#  message(STATUS SOURCE ${SOURCE} COMPILE_DEFINITIONS "${DISPLAY_COMPILE_DEFINITIONS}")
#endforeach()
target_compile_definitions(${TARGET} PRIVATE "${DISPLAY_COMPILE_DEFINITIONS}")

target_sources(${TARGET} PRIVATE ${${dir_LIB_SOURCES}})

if( "${GRAPHICS}" STREQUAL "X")
	find_package(Curses REQUIRED)
	target_include_directories(${TARGET} PRIVATE ${X11_INCLUDE_DIR} ${CURSES_INCLUDE_DIR})
	target_link_libraries(${TARGET} PUBLIC ${X11_LIBRARIES} ${CURSES_LIBRARY})
endif()
