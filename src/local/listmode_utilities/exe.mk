#
# $Id$
#

dir:=local/listmode_utilities

$(dir)_SOURCES = \
	lm_to_projdata_bootstrap.cxx \
	lm_to_projdata_with_MC.cxx \
	lm_fansums.cxx \
	find_motion_corrected_norm_factors.cxx \
	generate_headcurve.cxx

include $(WORKSPACE)/exe.mk
