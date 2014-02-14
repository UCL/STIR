#
#

dir := analytic/FBP3DRP
$(dir)_LIB_SOURCES:= \
	ColsherFilter.cxx  FBP3DRPReconstruction.cxx

#$(dir)_REGISTRY_SOURCES:= registries.cxx

include $(WORKSPACE)/lib.mk

