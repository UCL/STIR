# 
# $Id$
#

WORKSPACE=$(CURDIR)

include $(WORKSPACE)/Makefile_common



LIBSUBDIRECTORIES = \
	$(WORKSPACE)/buildblock \
	$(WORKSPACE)/IO \
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

ifeq ($(wildcard local/Makefile), local/Makefile)
__TEST_local_Makefile:=1
endif


.PHONY: $(EXESUBDIRECTORIES) all install clean alllocal alldistrib

# all is the default target (becomes it comes first)
# it depends 2 targets: the files in the distribution, and any local files
all: alldistrib alllocal

# alldistrib target makes all files in the distribution (and any local
# libraries that are linked in by local/Makefile_bblibs). It depends
# on the directories whose dependency rules will just 
# execute make in their relevant subdirectory.
# This way, when make fails in one subdirectory, this make will fail 
# as well
alldistrib: $(EXESUBDIRECTORIES)


alllocal:
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



