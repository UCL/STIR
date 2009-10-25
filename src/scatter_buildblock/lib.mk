
#
# $Id$
#

dir := scatter_buildblock
$(dir)_LIB_SOURCES:= \
	sample_scatter_points.cxx \
	single_scatter_estimate.cxx \
	single_scatter_integrals.cxx \
	scatter_detection_modelling.cxx \
        cached_single_scatter_integrals.cxx \
	scatter_estimate_for_one_scatter_point.cxx \
	upsample_and_fit_scatter_estimate.cxx \
	ScatterEstimationByBin.cxx 

#$(dir)_REGISTRY_SOURCES:= scatter_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk
