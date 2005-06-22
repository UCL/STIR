#
# $Id$
#
dir := local/SimSET

$(dir)_SOURCES :=  make_phg.c \
	conv_SimSET_STIR.cxx \
	conv_to_SimSET_image.cxx



include $(WORKSPACE)/exe.mk
