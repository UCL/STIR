#
# $Id$
#

dir := local/scatter_buildblock
$(dir)_LIB_SOURCES:= \
	sample_scatter_points.cxx \
	integral_scattpoint_det.cxx\
        cached_factors.cxx\
	cached_factors_2.cxx\
	scatter_estimate_for_one_scatter_point.cxx\
	scatter_estimate_for_none_scatter_point.cxx\
	scatter_estimate_for_two_scatter_points.cxx\
	scatter_estimate_for_all_scatter_points.cxx\
	scatter_viewgram.cxx \
	write_statistics.cxx

#$(dir)_REGISTRY_SOURCES:= local_scatter_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk
