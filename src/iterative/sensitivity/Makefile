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


${DEST}sensitivity: ${DEST}sensitivity.o ../OSMAPOSL/$(DEST)OSMAPOSLParameters.o \
	$(BB_LIBS) ${LIB_LOGLIK_BUILDINGBLOCK} $(BB_REGISTRIES)
	$(CXX) $(CFLAGS)-o $@ $< \
	  ../OSMAPOSL/$(DEST)OSMAPOSLParameters.o \
	  $(BB_REGISTRIES) \
	  ${LIB_LOGLIK_BUILDINGBLOCK} \
	  $(BB_LIBS) \
	  $(LINK_OPT) $(SYS_LIBS)

../OSMAPOSL/$(DEST)OSMAPOSLParameters.o:
	cd ../OSMAPOSL; $(MAKE) all


# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


.PRECIOUS: $(SOURCES:%.cxx=$(DEST)%.o) $(SOURCES:%.cxx=$(DEST)%)



