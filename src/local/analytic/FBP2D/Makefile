# MAKEFILE FOR FBP2D
#
# $Id$
#

# for fft.cxx
vpath %.cxx ../../buildblock:

WORKSPACE=$(CURDIR)/../../..

include $(WORKSPACE)/Makefile_common

SOURCES=   FBP2DMain.cxx \
 	fft.cxx RampFilter.cxx FBP2DReconstruction.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)


all: ${DEST}FBP2D

install: all
	cp $(DEST)FBP2D $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif

${DEST}FBP2D : ${OBJS} $(BB_LIBS) 
	$(CXX) $(CFLAGS)  -o $@ $(OBJS) \
	$(BB_LIBS)  $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)


clean :
	$(RM) $(OBJS) ${DEST}FBP3DRP



