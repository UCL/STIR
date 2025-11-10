#
#

set(dir analytic_DDSR2D) 

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	DDSR2DReconstruction.cxx
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_DDSR2D recon_buildblock IO )

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	DDSR2D.cxx
)

include(stir_exe_targets)
