#
# $Id$

include ../../Makefile_common

all: ${DEST}sensitivity

clean:
	$(RM) $(DEST)*

include ../../Makefile_bblibs

SOURCES = sensitivity.cxx

${DEST}sensitivity: ${DEST}sensitivity.o \
	$(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)-o $@ $< \
	${LIB_LOGLIK_BUILDINGBLOCK} \
	$(LIB_BUILDINGBLOCK) \
  	$(LIB_RECON_BUILDINGBLOCK) \
	 $(LINK_OPT) $(SYS_LIBS)


include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(SOURCES:%.cxx=$(DEST)%.o) $(SOURCES:%.cxx=$(DEST)%)



