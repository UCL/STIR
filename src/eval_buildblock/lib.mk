#
#

dir := eval_buildblock

$(dir)_LIB_SOURCES = \
  compute_ROI_values.cxx \
  ROIValues.cxx



#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk


