# 
# $Id$: 
#

WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common

LIBSUBDIRECTORIES = \
	$(WORKSPACE)/local/buildblock \
	$(WORKSPACE)/local/recon_buildblock \
	$(WORKSPACE)/local/Shape_buildblock \
	$(WORKSPACE)/local/eval_buildblock \
	$(WORKSPACE)/local/QH_library \
	$(WORKSPACE)/local/motion \
	$(WORKSPACE)/local/listmode	

#KT temp removal of QH_utilities as they don't link
EXESUBDIRECTORIES= \
        $(WORKSPACE)/local/utilities \
        $(WORKSPACE)/local/listmode_utilities \
	$(WORKSPACE)/local/analytic/FBP3DRP \
	$(WORKSPACE)/local/iterative/OSSPS 

SUBDIRECTORIES = $(LIBSUBDIRECTORIES) $(EXESUBDIRECTORIES)

.PHONY: $(EXESUBDIRECTORIES) all install clean

all: $(EXESUBDIRECTORIES)

$(EXESUBDIRECTORIES):
	$(MAKE) -C $@ all


install: all
	for subdir in $(EXESUBDIRECTORIES); do   \
	   echo "-------------------"; $(MAKE) -C $$subdir install; \
	done


clean: 
	for subdir in $(SUBDIRECTORIES); do \
	   echo "-------------------"; $(MAKE) -C $$subdir clean; \
	done;







