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


.PHONY: $(EXESUBDIRECTORIES) all install clean

# all is the default target (becomes it comes first)
# it depends on the directories whose dependency rules will just 
# execute make in their relevant subdirectory
# this way, when make fails in one subdirectory, this make will fail 
# as well
all: $(EXESUBDIRECTORIES)
ifeq ($(__TEST_local_Makefile),1)
	(cd local; $(MAKE) all);
endif


$(EXESUBDIRECTORIES):
	$(MAKE) -C $@ all



# install depends on all, such that the recursive make process
# fails if there's a problem in any of the subdirectories
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
ifeq ($(__TEST_local_Makefile),1)
	$(MAKE) -C local clean;
endif



