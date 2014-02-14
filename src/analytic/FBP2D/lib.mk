#
#

dir := analytic/FBP2D
$(dir)_LIB_SOURCES:= \
	RampFilter.cxx FBP2DReconstruction.cxx

#$(dir)_REGISTRY_SOURCES:= 

include $(WORKSPACE)/lib.mk

