#
#
set(dir utilities/UPENN)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

set(${dir_EXE_SOURCES} 
	conv_UPENN_projdata_to_STIR.cxx
)

include(stir_exe_targets)

target_link_libraries(conv_UPENN_projdata_to_STIR ${UPENN_libgeom}
    ${UPENN_liblor} ${UPENN_libimagio} ${UPENN_libimagio++} -lboost_system)
