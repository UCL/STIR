# $Id$
#
# Author Kris Thielemans
# Copyright 2004- $Date$ Hammersmith Imanet Ltd
# This file is part of STIR, and is distributed under the 
# terms of the GNU Lesser General Public Licence (LGPL) Version 2.1.
#
# This file will/can be included by a makefile skeleton in a subdirectory
# Requirements:
# the skeleton should set the following variables
#    dir must be set to the subdirectory name (relative to the location of Makefile)
#    $(dir)_TEST_SOURCES must be set to a list of .cxx and/or .c files (without path)
#    it must have a target run_tests_$(dir) (that depends on all_test_exes)
#    it must have a target run_interactive_tests_$(dir) (that depends on all_test_exes)
# Result:
# targets   build_tests_$(dir), clean_tests_$(dir)
# variables $(dir)_TEST_SOURCES (with $dir)/ prepended)
#           $(dir)_TEST_EXES, $(dir)_TEST_EXE_FILENAMES
# 
#
# Example for dir=SUBDIR
#
# make build_tests_SUBDIR will compile and link all executables
# make clean_tests_SUBDIR will remove all generated files
# make run_tests_SUBDIR will run tests for all executables

#$(warning including test.mk from $(dir))

$(dir)_TEST_SOURCES:= $(addprefix $(dir)/, $($(dir)_TEST_SOURCES))
$(dir)_TEST_EXES:= \
	$(patsubst %.cxx, $(DEST)%, $(filter %.cxx, $($(dir)_TEST_SOURCES))) \
	$(patsubst %.c, $(DEST)%, $(filter %.c, $($(dir)_TEST_SOURCES)))


$(dir)_TEST_EXE_FILENAMES := $(addsuffix $(EXE_SUFFIX), $($(dir)_TEST_EXES))

.PHONY: build_tests_$(dir) clean_tests_$(dir)  run_tests_$(dir) run_interactive_tests_$(dir)


# make sure make keeps the .o files
# otherwise it will be deleted
.PRECIOUS: $(patsubst %.cxx, $(DEST)%.o, $(filter %.cxx, $($(dir)_TEST_SOURCES))) 

build_tests_$(dir):  $($(dir)_TEST_EXES)

clean_tests_$(dir):
	rm -f $($(@:clean_tests_%=%)_TEST_EXE_FILENAMES)
	rm -f $(DEST)$(@:clean_tests_%=%)/*.[oP]


ifneq ($(MAKECMDGOALS:clean%=clean),clean)
  -include \
	$(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_TEST_SOURCES))) \
	$(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_TEST_SOURCES)))
endif

# set to some garbage such that we get an error when  the next skeleton forgets to set dir
dir := dir_not_set_in_test.mk
