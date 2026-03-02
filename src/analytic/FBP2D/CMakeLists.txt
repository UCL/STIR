#
#

set(dir analytic_FBP2D)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	RampFilter 
	FBP2DReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_FBP2D recon_buildblock IO )

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	FBP2D
)

include(stir_exe_targets)
