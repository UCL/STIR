#
#

set(dir analytic_DFM3D)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	DFM3DReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_DFM3D PUBLIC ${STIR_RECON_BUILDBLOCK_LIB})

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	DFM3D
)

include(stir_exe_targets)
