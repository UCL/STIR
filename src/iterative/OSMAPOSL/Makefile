#
# $Id$

include ../../Makefile_common

all: ${DEST}OSEMMain

clean:
	$(RM) $(DEST)*

include ../../Makefile_bblibs

SOURCES =  \
         OSMAPOSLParameters.cxx \
         OSMAPOSLReconstruction.cxx OSEMMain.cxx

${DEST}OSEMMain: ${DEST}OSEMMain.o \
          ${DEST}OSMAPOSLParameters.o ${DEST}OSMAPOSLReconstruction.o \
	$(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)-o $@ $< \
        ${DEST}OSMAPOSLParameters.o \
	${DEST}OSMAPOSLReconstruction.o \
	${LIB_LOGLIK_BUILDINGBLOCK} \
	$(LIB_BUILDINGBLOCK) \
  	$(LIB_RECON_BUILDINGBLOCK) \
	 $(LINK_OPT) $(SYS_LIBS)


include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(SOURCES:%.cxx=$(DEST)%.o) $(SOURCES:%.cxx=$(DEST)%)



