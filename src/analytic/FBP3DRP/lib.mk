#
# $Id$
#

dir := local/analytic/FBP3DRP
$(dir)_LIB_SOURCES:= \
	ColsherFilter.cxx  FBP3DRPReconstruction.cxx

#$(dir)_REGISTRY_SOURCES:= local_motion_registries.cxx

include $(WORKSPACE)/lib.mk

