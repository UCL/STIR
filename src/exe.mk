# $Id$
# This file will/can be included by a makefile skeleton in a subdirectory
# Requirements:
# the skeleton should set the following variables
#    dir must be the subdirectory (relative to the root)
#    $(dir)_SOURCES must be set to a list of .cxx or .c files
# Result:
# targets and some variables
#
# Example for dir=SUBDIR
#
# make SUBDIR will compile and link all executables
# make clean_SUBDIR will remove all generated files
# make install_SUBDIR will install all executables

#$(warning including exe.mk from $(dir))

$(dir)_SOURCES:= $(addprefix $(dir)/, $($(dir)_SOURCES))
$(dir)_EXES:= \
	$(patsubst %.cxx, $(DEST)%, $(filter %.cxx, $($(dir)_SOURCES))) \
	$(patsubst %.c, $(DEST)%, $(filter %.c, $($(dir)_SOURCES)))

.PHONY: $(dir) clean_$(dir) install_$(dir)

# make sure make keeps the .o file
# otherwise it will be deleted
.PRECIOUS: $(patsubst %.cxx, $(DEST)%.o, $(filter %.cxx, $($(dir)_SOURCES))) 

$(dir):  $($(dir)_EXES)

ifeq ($(SYSTEM),CYGWIN)
$(dir)_EXE_FILENAMES := $(addsuffix .exe, $($(dir)_EXES))
else
$(dir)_EXE_FILENAMES := $($(dir)_EXES)
endif

clean_$(dir):
	rm -f $($(@:clean_%=%)_EXE_FILENAMES)
	rm -f $(DEST)$(@:clean_%=%)/*.[oP]


install_$(dir): $(dir)
	cp $($(@:install_%=%)_EXE_FILENAMES) $(INSTALL_DIR)

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),clean_$(dir))
  -include \
	$(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_SOURCES)))) \
	$(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_SOURCES)))
endif
endif

# set to some garbage such that we get an error when  the next skeleton forgets to set dir
dir := dir_not_set_in_exe.mk
