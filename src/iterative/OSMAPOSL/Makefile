#
# $Id$
#
include ../../Makefile_common


SOURCES =  \
         OSMAPOSLParameters.cxx \
         OSMAPOSLReconstruction.cxx OSEMMain.cxx

EXES = \
	$(DEST)OSEMMain

all: $(EXES)

clean:
	$(RM) $(DEST)*


install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)


include ../../Makefile_bblibs


${DEST}OSEMMain: ${DEST}OSEMMain.o \
          ${DEST}OSMAPOSLParameters.o ${DEST}OSMAPOSLReconstruction.o \
	$(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)-o $@ $< \
        ${DEST}OSMAPOSLParameters.o \
	${DEST}OSMAPOSLReconstruction.o \
	${LIB_LOGLIK_BUILDINGBLOCK} \
	$(LIB_RECON_BUILDINGBLOCK) \
	$(LIB_BUILDINGBLOCK) \
	$(LINK_OPT) $(SYS_LIBS)



include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(SOURCES:%.cxx=$(DEST)%.o) $(SOURCES:%.cxx=$(DEST)%)



