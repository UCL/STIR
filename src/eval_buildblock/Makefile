#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common

all: $(DEST)libeval_buildblock.a

clean:
	$(RM) $(DEST)*


SOURCES = \
  compute_ROI_values.cxx \
  ROIValues.cxx




# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif

.PRECIOUS: $(DEST)*.o $(DEST)*.a


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

$(DEST)libeval_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


