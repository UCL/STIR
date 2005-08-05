#
# $Id$
#
dir := local/SimSET

$(dir)_SOURCES :=  make_phg.c \
	conv_SimSET_STIR.cxx \
	conv_to_SimSET_att_image.cxx \
	conv_to_SimSET_act_image.cxx



include $(WORKSPACE)/exe.mk
