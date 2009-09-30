#
# $Id$
#

dir := modelling_buildblock
$(dir)_LIB_SOURCES:= \
	KineticModel.cxx \
	PatlakPlot.cxx \
	ParametricDiscretisedDensity.cxx

$(dir)_REGISTRY_SOURCES:= modelling_registries.cxx

include $(WORKSPACE)/lib.mk

