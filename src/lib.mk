# $Id$
# This file will/can be included by a makefile skeleton in a subdirectory
# Requirements:
# the skeleton should set the following variables
#    dir must be the subdirectory (relative to the root)
#    $(dir)_LIBSOURCES must be set to a list of .cxx or .c files
#    $(dir)_REGISTRY_SOURCES must be set to a list of .cxx or .c files
# Result:
# targets and some variables
#

#$(warning including lib.mk from $(dir))

$(dir)_LIB_SOURCES:= $(addprefix $(dir)/, $($(dir)_LIB_SOURCES))
$(dir)_REGISTRY_SOURCES:= $(addprefix $(dir)/, $($(dir)_REGISTRY_SOURCES))
$(dir)_LIB_OBJS:=$(patsubst %.cxx, $(DEST)%.o, $(filter %.cxx, $($(dir)_LIB_SOURCES)))
$(dir)_LIB_OBJS+=$(patsubst %.c, $(DEST)%.o, $(filter %.c, $($(dir)_LIB_SOURCES)))
$(dir)_REGISTRY_OBJS:=$(patsubst %.cxx, $(DEST)%.o, $(filter %.cxx, $($(dir)_REGISTRY_SOURCES)))
$(dir)_REGISTRY_OBJS+=$(patsubst %.c, $(DEST)%.o, $(filter %.c, $($(dir)_REGISTRY_SOURCES)))

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),clean_$(dir))
  $(dir)_ALL_SOURCES:=$($(dir)_LIB_SOURCES) $($(dir)_REGISTRY_SOURCES)
  -include \
      $(patsubst %.cxx, $(DEST)%.P, $(filter %.cxx, $($(dir)_ALL_SOURCES))) \
      $(patsubst %.c, $(DEST)%.P, $(filter %.c, $($(dir)_ALL_SOURCES)))
endif
endif

.PRECIOUS: $($(dir)_REGISTRY_OBJS)

.PHONY: clean_$(dir)

# ideally we'd just have the following
#
#clean_$(dir): 
#	rm -f $(DEST)$(dir)/*.[oP]
# this doesn't work, because $(dir) in the execution line is evaluated
# at the end of all Makefile reading. By that time, $(dir) will be something else.
# the following work-around by Paul D Smith does the trick

clean_$(dir):
	rm -f $(DEST)$(@:clean_%=%)/*.[oP]



dir := dir_not_set_in_lib.mk
