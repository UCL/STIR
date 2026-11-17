
# cmake helper file for building STIR.
set(dir iterative_NESTPOSMAPOSL)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

# include(stir_lib_target)

set (dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
   NESTPOSMAPOSL.cxx
)

include(stir_exe_targets)

if (STIR_MPI)
  SET_PROPERTY(TARGET NESTPOSMAPOSL PROPERTY LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endif()

# target_link_libraries(iterative_NESTPOSMAPOSL PUBLIC ${STIR_RECON_BUILDBLOCK_LIB})
