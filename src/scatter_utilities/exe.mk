#
# $Id$
#
dir := local/scatter

$(dir)_SOURCES :=  estimate_single_scatter.cxx \
	scale_single_scatter.cxx



include $(WORKSPACE)/exe.mk
