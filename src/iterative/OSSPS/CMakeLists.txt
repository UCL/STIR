#
# $Id$
#

set(dir iterative_OSSPS)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	OSSPSReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

add_library(${dir} ${${dir_LIB_SOURCES}}    )

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	OSSPS
)


include(../../stir_exe_targets.cmake)
