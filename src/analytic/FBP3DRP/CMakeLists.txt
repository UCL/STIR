#
#

set(dir analytic_FBP3DRP)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	ColsherFilter.cxx
	FBP3DRPReconstruction.cxx
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_FBP3DRP PUBLIC analytic_FBP2D recon_buildblock PUBLIC fmt )

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	FBP3DRP.cxx
)

include(stir_exe_targets)

