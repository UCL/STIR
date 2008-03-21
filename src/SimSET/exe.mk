#
# $Id$
#
dir := SimSET

$(dir)_SOURCES :=  \
	conv_SimSET_projdata_to_STIR.cxx \
	conv_to_SimSET_att_image.cxx

include $(WORKSPACE)/exe.mk
