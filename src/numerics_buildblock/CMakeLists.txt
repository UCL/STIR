#
#
set(dir numerics_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  fourier
  determinant
)

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include(stir_lib_target)

target_link_libraries(${dir} buildblock)
