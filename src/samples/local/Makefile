# 
# $Id$: 
#

WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common

LIBSUBDIRECTORIES = \
	$(WORKSPACE)/local/buildblock \
	$(WORKSPACE)/local/QH_library

EXESUBDIRECTORIES= \
        $(WORKSPACE)/local/QH_utilities

SUBDIRECTORIES = $(LIBSUBDIRECTORIES) $(EXESUBDIRECTORIES)

.PHONY: $(EXESUBDIRECTORIES) all install clean

all: $(EXESUBDIRECTORIES)

$(EXESUBDIRECTORIES):
	$(MAKE) -C $@ all


install: all
	for subdir in $(EXESUBDIRECTORIES); do   \
	   echo "-------------------"; $(MAKE) -C $$subdir install; \
	done
ifeq ($(__TEST_local_Makefile),1)
	(cd local; $(MAKE) install);
endif


clean: 
	for subdir in $(SUBDIRECTORIES); do \
	   echo "-------------------"; $(MAKE) -C $$subdir clean; \
	done;
