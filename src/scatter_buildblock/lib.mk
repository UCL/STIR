#
# $Id$
#

dir := local/scatter_buildblock
$(dir)_LIB_SOURCES:= \
	sample_scatter_points.cxx \
	integral_scattpoint_det.cxx\ 
	cross_section.cxx\
	scatter_estimate_for_one_scatter_point.cxx\
	scatter_estimate_for_all_scatter_points.cxx\
        scatter_viewgram.cxx

#$(dir)_REGISTRY_SOURCES:= local_scatter_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk
