#
# $Id$
#

set(dir analytic_FBP2D)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	RampFilter 
	FBP2DReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

add_library(${dir} ${${dir_LIB_SOURCES}}    )

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	FBP2D
)

include(../../stir_exe_targets.cmake)
