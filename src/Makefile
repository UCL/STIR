# 
# $Id$: $Date$
#
include Makefile_common

LIBSUBDIRECTORIES = \
	$(WORKSPACE)/buildblock \
	$(WORKSPACE)/recon_buildblock \
	$(WORKSPACE)/display \
	$(WORKSPACE)/iterative/LogLik_buildblock 

EXESUBDIRECTORIES=
        $(WORKSPACE)/iterative/sensitivity \
	$(WORKSPACE)/iterative/OSMAPOSL \
	$(WORKSPACE)/utilities \
	$(WORKSPACE)/recon_test \
	$(WORKSPACE)/test 

SUBDIRECTORIES = $(LIBSUBDIRECTORIES) $(EXESUBDIRECTORIES)

MAKE_ALL = $(MAKE) all

all: 
	for subdir in $(EXESUBDIRECTORIES); do   \
	  	(echo "------------------");\
		(cd $$subdir; $(MAKE_ALL)); \
	done



install:
	for subdir in $(EXESUBDIRECTORIES); do   \
	  	(echo "------------------");\
		(cd $$subdir; $(MAKE) install); \
	done

clean: 
#	echo -n "Are you sure to clean up all the directories ? (y/n)"
#	set resp = $<
#	if ($resp == 'y') then
		for subdir in $(SUBDIRECTORIES); do \
	  		(echo \rm -rf $$subdir/$(DEST)); \
			(cd $$subdir; \rm -r $(DEST));\
		done; \
#	else
#		(echo "No clean up ");
#	fi


.PHONY: all clean
