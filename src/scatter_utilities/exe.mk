#
# $Id$
#
dir := local/scatter

$(dir)_SOURCES :=  estimate_single_scatter.cxx/
                   write_statistics


include $(WORKSPACE)/exe.mk
