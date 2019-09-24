
set(dir iterative_KOSMAPOSL)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	KOSMAPOSLReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	KOSMAPOSL
)

include(stir_exe_targets)

if (STIR_MPI)
  SET_PROPERTY(TARGET KOSMAPOSL PROPERTY LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endif()

target_link_libraries(iterative_KOSMAPOSL  recon_buildblock )

