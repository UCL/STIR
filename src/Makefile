# 
# $Id$: 
#
include Makefile_common



LIBSUBDIRECTORIES = \
	$(WORKSPACE)/buildblock \
	$(WORKSPACE)/recon_buildblock \
	$(WORKSPACE)/display \
	$(WORKSPACE)/iterative/LogLik_buildblock 

EXESUBDIRECTORIES= \
        $(WORKSPACE)/iterative/sensitivity \
	$(WORKSPACE)/iterative/OSMAPOSL \
	$(WORKSPACE)/utilities \
	$(WORKSPACE)/recon_test \
	$(WORKSPACE)/test 

SUBDIRECTORIES = $(LIBSUBDIRECTORIES) $(EXESUBDIRECTORIES)


__TEST_local_Makefile:=$(shell if [ -e local/Makefile ]; then echo 1;  fi)



all: 
	for subdir in $(EXESUBDIRECTORIES); do   \
	  	(echo "------------------");\
		(cd $$subdir; $(MAKE) all); \
	done;
ifeq ($(__TEST_local_Makefile),1)
	(cd local; $(MAKE) all);
endif


install:
	for subdir in $(EXESUBDIRECTORIES); do   \
	  	(echo "------------------");\
		(cd $$subdir; $(MAKE) install); \
	done
ifeq ($(__TEST_local_Makefile),1)
	(cd local; $(MAKE) install);
endif


clean: 
	for subdir in $(SUBDIRECTORIES); do 
	  	(echo \rm -rf $$subdir/$(DEST)); 
		(cd $$subdir; \rm -r $(DEST));
	done; 
ifeq ($(__TEST_local_Makefile),1)
	(cd local; $(MAKE) all);
endif


.PHONY: all clean install
