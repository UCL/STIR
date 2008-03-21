#
# $Id$
#
dir := local/SimSET

$(dir)_SOURCES :=  make_phg.c \
	conv_to_SimSET_act_image.cxx



include $(WORKSPACE)/exe.mk
