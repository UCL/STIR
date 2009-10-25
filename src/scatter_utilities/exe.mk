#
# $Id$
#
dir := scatter_utilities

$(dir)_SOURCES := \
	estimate_scatter.cxx \
	create_tail_mask_from_ACFs.cxx \
	upsample_and_fit_single_scatter.cxx

include $(WORKSPACE)/exe.mk
