#
#

set(dir analytic_FBP2D)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	RampFilter.cxx
	FBP2DReconstruction.cxx
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_FBP2D PUBLIC recon_buildblock IO fmt::fmt)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	FBP2D.cxx
)

include(stir_exe_targets)
