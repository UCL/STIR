#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common

SOURCES = bcktest.cxx fwdtest.cxx #zoomtest.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)



# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif

include $(WORKSPACE)/Makefile_bblibs

