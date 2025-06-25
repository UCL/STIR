#
#

set(dir analytic_SRT2DSPECT)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	SRT2DSPECTReconstruction.cxx
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)
target_link_libraries(analytic_SRT2DSPECT PRIVATE fmt PUBLIC recon_buildblock IO)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	SRT2DSPECT.cxx
)

include(stir_exe_targets)
