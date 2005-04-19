#
# $Id$
#
dir := local/SimSET

$(dir)_SOURCES :=  make_phg.c \
	conv_SimSET_STIR.cxx



include $(WORKSPACE)/exe.mk
