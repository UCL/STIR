#
# $Id:
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common

all: $(DEST)libShape_buildblock.a $(DEST)Shape_buildblock_registries.o

clean:
	$(RM) $(DEST)*

SOURCES = \
  Shape3D.cxx \
  DiscretisedShape3D.cxx \
  Shape3DWithOrientation.cxx \
  Ellipsoid.cxx \
  EllipsoidalCylinder.cxx
  


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


$(DEST)libShape_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


