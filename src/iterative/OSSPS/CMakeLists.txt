
set(dir iterative_OSSPS)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	OSSPSReconstruction
)

#$(dir)_REGISTRY_SOURCES:= 

include(stir_lib_target)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	OSSPS
)


include(stir_exe_targets)

if (STIR_MPI)
  SET_PROPERTY(TARGET OSSPS PROPERTY LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endif()

target_link_libraries(iterative_OSSPS recon_buildblock )
