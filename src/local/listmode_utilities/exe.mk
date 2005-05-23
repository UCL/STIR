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
	get_singles_info.cxx \
	change_lm_time_tags.cxx \
	scan_singles_file.cxx \
	print_sgl_values.cxx \
	rebin_singles_file.cxx


include $(WORKSPACE)/exe.mk
