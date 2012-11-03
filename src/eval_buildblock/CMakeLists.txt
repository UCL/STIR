#
# $Id$
#

set(dir eval_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  compute_ROI_values
  ROIValues
)


#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include(../stir_lib_target.cmake)

target_link_libraries(eval_buildblock buildblock )


