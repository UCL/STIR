# $Id$
# This file will/can be included by a makefile skeleton in a subdirectory
# Requirements:
# the skeleton should set the following variables
#    dir must be the subdirectory (relative to the root)
#    $(dir)_TEST_SOURCES must be set to a list of .cxx or .c files
#    it must have a target run_tests_$(dir) (that depends on all_test_exes)
# Result:
# targets and some variables
#
# Example for dir=SUBDIR
#
# make SUBDIR will compile and link all executables
# make clean_SUBDIR will remove all generated files
# make run_test_SUBDIR will install all executables

#$(warning including exe.mk from $(dir))

$(dir)_TEST_SOURCES:= $(addprefix $(dir)/, $($(dir)_TEST_SOURCES))
$(dir)_TEST_EXES:= \
	$(patsubst %.cxx, $(DEST)%, $(filter %.cxx, $($(dir)_TEST_SOURCES))) \
	$(patsubst %.c, $(DEST)%, $(filter %.c, $($(dir)_TEST_SOURCES)))


ifeq ($(SYSTEM),CYGWIN)
$(dir)_TEST_EXE_FILENAMES := $(addsuffix .exe, $($(dir)_TEST_EXES))
else
$(dir)_TEST_EXE_FILENAMES := $($(dir)_TEST_EXES)
endif

.PHONY: $(dir) clean_$(dir)  $(dir)_run_tests $(dir)_run_interactive_tests 


# make sure make keeps the .o file
# otherwise it will be deleted
.PRECIOUS: $(patsubst %.cxx, $(DEST)%.o, $(filter %.cxx, $($(dir)_TEST_SOURCES))) 

$(dir):  $($(dir)_TEST_EXES)

clean_$(dir):
	rm -f $($(@:clean_%=%)_TEST_EXE_FILENAMES)
	rm -f $(DEST)$(@:clean_%=%)/*.[oP]



ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),clean_$(dir))
  -include \
	$(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_TEST_SOURCES))) \
	$(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_TEST_SOURCES)))
endif
endif

# set to some garbage such that we get an error when  the next skeleton forgets to set dir
dir := dir_not_set_in_test.mk
