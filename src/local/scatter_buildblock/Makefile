#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libscatter_buildblock.a

clean:
	$(RM) $(DEST)*

SOURCES = \
	sample_scatter_points.cxx \
	integral_scattpoint_det.cxx \
	scatter_estimate_for_one_scatter_point.cxx\
	scatter_estimate_for_all_scatter_points.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif


.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libscatter_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@



