# 
# $Id$: $Date$
#
include Makefile_common

SUBDIRECTORY = \
	$(WORKSPACE)/buildblock \
	$(WORKSPACE)/recon_buildblock \
	$(WORKSPACE)/display \
	$(WORKSPACE)/iterative/LogLik_buildblock \
        $(WORKSPACE)/iterative/sensitivity \
	$(WORKSPACE)/iterative/OSMAPOSL \
	$(WORKSPACE)/utilities \
	$(WORKSPACE)/recon_test \
	$(WORKSPACE)/test 

MAKE_ALL = $(MAKE) all

all: 
	for subdir in $(SUBDIRECTORY); do   \
	  	(echo "------------------");\
		(cd $$subdir; $(MAKE_ALL)); \
	done




clean: 
#	echo -n "Are you sure to clean up all the directories ? (y/n)"
#	set resp = $<
#	if ($resp == 'y') then
		for subdir in $(SUBDIRECTORY); do \
	  		(echo \rm -rf $$subdir/$(DEST)); \
			(cd $$subdir; \rm -r $(DEST));\
		done; \
#	else
#		(echo "No clean up ");
#	fi


.PHONY: all clean
