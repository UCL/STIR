# MAKEFILE FOR FBP3DRP 
#
# $Id$
#

vpath %.cxx ../../buildblock:../FBP2D

WORKSPACE=$(CURDIR)/../../..

include $(WORKSPACE)/Makefile_common

SOURCES_FBP3DRP= ColsherFilter.cxx  FBP3DRPReconstruction.cxx FBP3DRPParameters.cxx FBP3DRPMain.cxx \
fft.cxx merge_direct_sinos.cxx RampFilter.cxx FBP2DReconstruction.cxx

OBJS_FBP3DRP = $(SOURCES_FBP3DRP:%.cxx=$(DEST)%.o)


all: ${DEST}FBP3DRP

install: all
	cp $(DEST)FBP3DRP $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
include $(SOURCES_FBP3DRP:%.cxx=${DEST}%.d)
endif

${DEST}FBP3DRP : ${OBJS_FBP3DRP} $(BB_LIBS) 
	$(CXX) $(CFLAGS)  -o $@ $(OBJS_FBP3DRP) \
	$(BB_LIBS)  $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)


clean :
	$(RM) $(OBJS_FBP3DRP) ${DEST}FBP3DRP



