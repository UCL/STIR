#
# Author Kris Thielemans
# Copyright 2004- 2004 Hammersmith Imanet Ltd
# This file is part of STIR, and is distributed under the 
# terms of the GNU Lesser General Public Licence (LGPL) Version 2.1.
#
# This file will/can be included by a makefile skeleton in a subdirectory
# Requirements:
# the skeleton should set the following variables
#    dir must be the subdirectory (relative to the root)
#    $(dir)_LIB_SOURCES must be set to a list of .cxx or .c files (without path)
#    $(dir)_REGISTRY_SOURCES must be set to a list of .cxx or .c files (without path)
# Result:
# targets clean_lib_$(dir) build_lib_$(dir)
# variables $(dir)_LIB_SOURCES, $(dir)_REGISTRY_OBJS (both with $dir)/ prepended)
#           $(dir)_LIB_OBJS


#$(warning including lib.mk from $(dir))

.PHONY: clean_lib_$(dir) build_lib_$(dir)


$(dir)_LIB_SOURCES:= $(addprefix $(dir)/, $($(dir)_LIB_SOURCES))
$(dir)_REGISTRY_SOURCES:= $(addprefix $(dir)/, $($(dir)_REGISTRY_SOURCES))
$(dir)_LIB_OBJS:= \
	$(patsubst %.cxx, $(DEST)%${O_SUFFIX}, $(filter %.cxx, $($(dir)_LIB_SOURCES))) \
	$(patsubst %.c, $(DEST)%${O_SUFFIX}, $(filter %.c, $($(dir)_LIB_SOURCES)))
$(dir)_REGISTRY_OBJS:= \
	$(patsubst %.cxx, $(DEST)%${O_SUFFIX}, $(filter %.cxx, $($(dir)_REGISTRY_SOURCES))) \
	$(patsubst %.c, $(DEST)%${O_SUFFIX}, $(filter %.c, $($(dir)_REGISTRY_SOURCES)))

build_lib_$(dir):  $($(dir)_LIB_OBJS) $($(dir)_REGISTRY_OBJS)

ifneq ($(MAKECMDGOALS:clean%=clean),clean)
  $(dir)_ALL_SOURCES:=$($(dir)_LIB_SOURCES) $($(dir)_REGISTRY_SOURCES)
  -include \
      $(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_ALL_SOURCES))) \
      $(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_ALL_SOURCES)))
endif

.PRECIOUS: $($(dir)_REGISTRY_OBJS)

# ideally we'd just have the following
#
#clean_lib_$(dir): 
#	rm -f $(DEST)$(dir)/*.[oP]
# this doesn't work, because $(dir) in the execution line is evaluated
# at the end of all Makefile reading. By that time, $(dir) will be something else.
# the following work-around by Paul D Smith does the trick

clean_lib_$(dir):
	rm -f $(DEST)$(@:clean_lib_%=%)/*.[P]
	rm -f $(DEST)$(@:clean_lib_%=%)/*$(O_SUFFIX)



# set to some garbage such that we get an error when  the next skeleton forgets to set dir
dir := dir_not_set_in_lib.mk
