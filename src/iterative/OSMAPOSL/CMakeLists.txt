#
# $Id$
#

set(dir iterative_OSMAPOSL)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	OSMAPOSLReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

add_library(${dir} ${${dir_LIB_SOURCES}}    )

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	OSMAPOSL
)

include(../../stir_exe_targets.cmake)


