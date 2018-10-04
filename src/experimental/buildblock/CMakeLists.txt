set(dir local_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	Quaternion.cxx 
	SeparableGaussianArrayFilter.cxx
	SeparableGaussianImageFilter.cxx
)

include(stir_lib_target)

target_link_libraries(${dir} buildblock)

