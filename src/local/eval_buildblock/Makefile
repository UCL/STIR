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



OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libeval_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


