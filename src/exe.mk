#
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
#    $(dir)_SOURCES must be set to a list of .cxx and/or .c files (without path)
# Result:
# targets $(dir) clean_exes_$(dir) install_exes_$(dir), uninstall_exes_$(dir)
#    and for each file in $(dir)_SOURCES there will be a (phony) target
#    $(dir)/file (without extension) set to depend on $(DEST)$(dir)/file
#    (see below)
# variables $(dir)_SOURCES (with $dir)/ prepended)
#           $(dir)_EXES, $(dir)_EXE_FILENAMES
#
# Example for dir=SUBDIR
#
# make SUBDIR              -> will compile and link all executables
# make clean_exes_SUBDIR   -> will remove all generated files
# make install_exes_SUBDIR      -> will install all executables
# make $(dir)/file         -> will compile and link single file
#                             (there is no need to use the $(DEST) prefix)

$(dir)_SOURCES:= $(addprefix $(dir)/, $($(dir)_SOURCES))
$(dir)_EXES:= \
	$(patsubst %.cxx, $(DEST)%, $(filter %.cxx, $($(dir)_SOURCES))) \
	$(patsubst %.c, $(DEST)%, $(filter %.c, $($(dir)_SOURCES)))
$(dir)_EXES_without_DEST:= \
	$(patsubst %.cxx, %, $(filter %.cxx, $($(dir)_SOURCES))) \
	$(patsubst %.c, %, $(filter %.c, $($(dir)_SOURCES)))

.PHONY: $(dir) clean_exes_$(dir) install_exes_$(dir) uninstall_exes_$(dir)

# make sure make keeps the .o files
# otherwise they will be deleted
.PRECIOUS: $(patsubst %.cxx, $(DEST)%$(O_SUFFIX), $(filter %.cxx, $($(dir)_SOURCES))) 

$(dir):  $($(dir)_EXES)

# trick (from the GNU make manual) to a target for every file which just
# depends on $(DEST)/file. The advantage for the user is that she doesn't
# have to type $(DEST) explictly anymore

define PROGRAM_template
$(1): $(DEST)$(1)

.PHONY: $(1) 
endef

$(foreach prog,$($(dir)_EXES_without_DEST),$(eval $(call PROGRAM_template,$(prog))))


# set up variable for clean_exes
$(dir)_EXE_FILENAMES := $(addsuffix $(EXE_SUFFIX), $($(dir)_EXES))


# note: see lib.mk for explanation for the $(@:...) trick
# it really is just a way to get $(dir) at build-time
clean_exes_$(dir):
	rm -f $($(@:clean_exes_%=%)_EXE_FILENAMES)
	rm -f $(DEST)$(@:clean_exes_%=%)/*.[P]
	rm -f $(DEST)$(@:clean_exes_%=%)/*$(O_SUFFIX)



install_exes_$(dir): $(dir)
	mkdir -p $(INSTALL_EXE_DIR)
	if [ ! -z "$($(@:install_exes_%=%)_EXE_FILENAMES)" ]; then $(INSTALL) $($(@:install_exes_%=%)_EXE_FILENAMES) $(INSTALL_EXE_DIR); fi

uninstall_exes_$(dir):
	$(RM) $(addprefix $(INSTALL_EXE_DIR)/, $(notdir $($(@:uninstall_exes_%=%)_EXE_FILENAMES)))

ifneq ($(MAKECMDGOALS:clean%=clean),clean)
  -include \
	$(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_SOURCES))) \
	$(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_SOURCES)))
endif

# set to some garbage such that we get an error when  the next skeleton forgets to set dir
dir := dir_not_set_in_exe.mk
