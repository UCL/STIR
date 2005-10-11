#
# $Id$
#
dir := local/scatter

$(dir)_SOURCES :=  estimate_single_scatter.cxx \
	estimate_scatter.cxx \
	scale_single_scatter.cxx

#	estimate_single_scatter_splitted.cxx \

include $(WORKSPACE)/exe.mk
