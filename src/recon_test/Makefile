#
# $Id$: $Date$
#
include ../Makefile_common

SOURCES = bcktest.cxx fwdtest.cxx #zoomtest.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

clean:
	$(RM) $(DEST)*

EXES = \
        $(DEST)sensitivity


install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)

include $(SOURCES:%.cxx=$(DEST)%.d)

include ../Makefile_bblibs

