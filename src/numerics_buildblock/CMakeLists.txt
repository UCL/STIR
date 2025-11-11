#
#
set(dir numerics_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  fourier.cxx
  determinant.cxx
)

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

target_sources(${STIR_BUILDBLOCK_LIB} PRIVATE ${${dir_LIB_SOURCES}})
