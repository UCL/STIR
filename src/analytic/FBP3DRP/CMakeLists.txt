#
# $Id$
#

set(dir analytic_FBP3DRP)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	ColsherFilter 
	FBP3DRPReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

include(../../stir_lib_target.cmake)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	FBP3DRP
)

include(../../stir_exe_targets.cmake)

