# 
# $Id$: 
#
include $(WORKSPACE)/Makefile_common

LIBSUBDIRECTORIES = \
	$(WORKSPACE)/local/buildblock \
	$(WORKSPACE)/local/QH_library

EXESUBDIRECTORIES= \
        $(WORKSPACE)/local/QH_utilities

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
