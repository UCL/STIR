#
# $Id$
#
dir := local/scatter

$(dir)_SOURCES := \
	estimate_scatter.cxx \
	scale_single_scatter.cxx \
	create_tail_mask_from_ACFs.cxx

#	estimate_single_scatter_splitted.cxx \

include $(WORKSPACE)/exe.mk
