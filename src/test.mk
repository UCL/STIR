#
# Author Kris Thielemans
# Copyright 2004- 2009 Hammersmith Imanet Ltd
#   This file is part of STIR.
#
#   This file is free software; you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation; either version 2.1 of the License, or
#   (at your option) any later version.
#
#   This file is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
# This file will/can be included by a makefile skeleton in a subdirectory
# Requirements:
# the skeleton should set the following variables
#    dir must be set to the subdirectory name (relative to the location of Makefile)
#    $(dir)_TEST_SOURCES must be set to a list of .cxx and/or .c files (without path)
#         that will be used for non-interactive tests
#    $(dir)_TEST_INTERACTIVE_SOURCES is similar for interactive tests
#    it must have a target run_$(dir)/some_test_exe 
#          for any 'some_test_exe' that needs command line arguments 
#	   or other non-standard command line
#          Ideally, all your run_$(dir)/some_test_exe targets depend on PHONY_TARGET (see below)
#    similarly, it must have a target run_interactive_$(dir)/some_test_exe
# Result:
# targets   build_tests_$(dir), clean_tests_$(dir)
#           run_tests_$(dir), run_interactive_tests_$(dir)
#	    run_$(dir)/file for any file in $(dir)_TEST_SOURCES
#           $(dir)/file for any file in $(dir)_TEST_SOURCES (which depends on $(DEST)$(dir)/file)
#           (see below)
# some variables (but you shouldn't count on those remaining the same)
# 
#
# Example for dir=SUBDIR
#
# make build_tests_SUBDIR will compile and link all executables
# make clean_tests_SUBDIR will remove all generated files
# make run_tests_SUBDIR will run tests for all (non-interactive) executables
# make run_SUBDIR/somefile will run a single test
# make SUBDIR/somefile will build a single test

#$(warning including test.mk from $(dir))


$(dir)_RUN_TEST_TARGETS:= \
	$(patsubst %.cxx, run_$(dir)/%, $(filter %.cxx, $($(dir)_TEST_SOURCES))) \
	$(patsubst %.c, run_$(dir)/%, $(filter %.c, $($(dir)_TEST_SOURCES)))
$(dir)_RUN_INTERACTIVE_TEST_TARGETS:= \
	$(patsubst %.cxx, run_$(dir)/%, $(filter %.cxx, $($(dir)_INTERACTIVE_TEST_SOURCES))) \
	$(patsubst %.c, run_$(dir)/%, $(filter %.c, $($(dir)_INTERACTIVE_TEST_SOURCES)))


run_tests_$(dir):  $($(dir)_RUN_TEST_TARGETS)

run_interactive_tests_$(dir):  $($(dir)_RUN_INTERACTIVE_TEST_TARGETS)

# default rule for run_$(dir)/some_test_exe
# This rule just runs the corresponding executable
run_$(dir)/% : $(DEST)$(dir)/% PHONY_TARGET
	$<
# Notes on above rule:
# Ideally, we would make all targets in $($(dir)_RUN_TEST_TARGETS) phony.
# However, make does not look for built-in rules for phony targets.
# So, instead, we let them depend on a phony target. Sorry.
#
# For some reason, I need $(dir) in there. The following does not work
# run_% : $(DEST)% PHONY_TARGET
# (probably because of the /)

.PHONY: PHONY_TARGET

# variables for the build* and clean* targets
$(dir)_ALL_TEST_SOURCES:= $($(dir)_TEST_SOURCES)  $($(dir)_TEST_INTERACTIVE_SOURCES)
$(dir)_ALL_TEST_SOURCES:= $(addprefix $(dir)/, $($(dir)_ALL_TEST_SOURCES))

$(dir)_TEST_OBJS:= \
	$(patsubst %.cxx, $(DEST)%$(O_SUFFIX), $(filter %.cxx, $($(dir)_ALL_TEST_SOURCES))) \
	$(patsubst %.c, $(DEST)%$(O_SUFFIX), $(filter %.c, $($(dir)_ALL_TEST_SOURCES)))
$(dir)_TEST_EXES:= \
	$(patsubst %$(O_SUFFIX), %,  $($(dir)_TEST_OBJS))
$(dir)_TEST_OBJS_without_DEST:= \
	$(patsubst %.cxx, %$(O_SUFFIX), $(filter %.cxx, $($(dir)_ALL_TEST_SOURCES))) \
	$(patsubst %.c, %$(O_SUFFIX), $(filter %.c, $($(dir)_ALL_TEST_SOURCES)))
$(dir)_TEST_EXES_without_DEST:= \
	$(patsubst %$(O_SUFFIX), %,  $($(dir)_TEST_OBJS_without_DEST))

$(dir)_TEST_EXE_FILENAMES := $(addsuffix $(EXE_SUFFIX), $($(dir)_TEST_EXES))

.PHONY: build_tests_$(dir) clean_tests_$(dir)  run_tests_$(dir) run_interactive_tests_$(dir)


# make sure 'make' keeps the .o files
# otherwise they will be deleted
.PRECIOUS: $($(dir)_ALL_TEST_OBJS)) 

# trick (from the GNU make manual) to define a target for every file which just
# depends on $(DEST)/file. The advantage for the user is that she doesn't
# have to type $(DEST) explictly anymore

define PROGRAM_template
$(1): $(DEST)$(1)

.PHONY: $(1) 
endef

$(foreach prog,$($(dir)_TEST_EXES_without_DEST),$(eval $(call PROGRAM_template,$(prog))))

build_tests_$(dir):  $($(dir)_TEST_EXES)

clean_tests_$(dir):
	rm -f $($(@:clean_tests_%=%)_TEST_EXE_FILENAMES)
	rm -f $($(@:clean_tests_%=%)_TEST_OBJS)
	rm -f $(DEST)$(@:clean_tests_%=%)/*.P 


ifneq ($(MAKECMDGOALS:clean%=clean),clean)
  -include \
	$(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_ALL_TEST_SOURCES))) \
	$(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_ALL_TEST_SOURCES)))
endif

# set to some garbage such that we get an error when  the next skeleton forgets to set dir
dir := dir_not_set_in_test.mk
