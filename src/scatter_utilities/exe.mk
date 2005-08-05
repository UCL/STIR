#
# $Id$
#
dir := local/scatter

$(dir)_SOURCES :=  estimate_single_scatter.cxx \
	estimate_single_scatter_splitted.cxx \
	scale_single_scatter.cxx

include $(WORKSPACE)/exe.mk
