#
# $Id$
#
WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common

SOURCES = sensitivity.cxx

EXES = \
	$(DEST)sensitivity

all: $(EXES)

clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs


${DEST}sensitivity: ${DEST}sensitivity.o \
	$(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)-o $@ $< \
	${LIB_LOGLIK_BUILDINGBLOCK} \
	$(LIB_RECON_BUILDINGBLOCK) \
	$(LIB_BUILDINGBLOCK) \
	$(LINK_OPT) $(SYS_LIBS)


# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


.PRECIOUS: $(SOURCES:%.cxx=$(DEST)%.o) $(SOURCES:%.cxx=$(DEST)%)



