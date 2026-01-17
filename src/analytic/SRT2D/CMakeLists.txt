#
#

set(dir analytic_SRT2D)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	SRT2DReconstruction.cxx
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_SRT2D PUBLIC ${STIR_RECON_BUILDBLOCK_LIB})

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	SRT2D.cxx
)

include(stir_exe_targets)
