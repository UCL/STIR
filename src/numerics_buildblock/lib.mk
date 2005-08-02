#
# $Id$
#
dir:=numerics_buildblock

$(dir)_LIB_SOURCES := \
  fourier.cxx \
  determinant.cxx

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk


