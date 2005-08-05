#
# $Id$
#

dir := local/scatter_buildblock
$(dir)_LIB_SOURCES:= \
	sample_scatter_points.cxx \
	integral_scattpoint_det.cxx \
        cached_factors.cxx \
	cached_factors_2.cxx \
	scatter_estimate_for_one_scatter_point.cxx \
	scatter_estimate_for_none_scatter_point.cxx \
	scatter_estimate_for_two_scatter_points.cxx \
	scatter_estimate_for_all_scatter_points.cxx \
	scatter_viewgram.cxx \
	scatter_estimate_for_all_scatter_points_splitted.cxx \
	scatter_estimate_for_two_scatter_points_splitted.cxx \
	scatter_viewgram_splitted.cxx \
	write_statistics.cxx \
	scale_factors_per_sinogram.cxx \
	att_estimate_for_no_scatter.cxx \
	estimate_att_viewgram.cxx \
	scale_scatter_per_sinogram.cxx


#$(dir)_REGISTRY_SOURCES:= local_scatter_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk
