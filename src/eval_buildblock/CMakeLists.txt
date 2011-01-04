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

add_library(${dir} ${${dir_LIB_SOURCES}}    )


