#
# $Id$
#

dir:=local/listmode_utilities

$(dir)_SOURCES = \
	lm_to_projdata_bootstrap.cxx \
	lm_to_projdata_with_MC.cxx \
	lm_fansums.cxx \
	generate_headcurve.cxx \
	list_time_events.cxx \
	change_lm_time_tags.cxx

include $(WORKSPACE)/exe.mk
