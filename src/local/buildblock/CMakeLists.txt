set(dir local_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	ML_norm.cxx 
	Quaternion.cxx 
)

include(${CMAKE_SOURCE_DIR}/stir_lib_target.cmake)

target_link_libraries(${dir} buildblock)

