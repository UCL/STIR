#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libmotion.a $(DEST)local_motion_registries.o

clean:
	$(RM) $(DEST)*

SOURCES = \
	Polaris_MT_File.cxx \
	RegisteredObject.cxx \
	RigidObject3DMotion.cxx\
	RigidObject3DMotionFromPolaris.cxx \
	RigidObject3DTransformation.cxx \
	object_3d_transform_image.cxx


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
  include  $(DEST)local_motion_registries.d
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
  -include  $(DEST)local_motion_registries.P
endif
endif


.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libmotion.a: $(OBJS)
	ar r $@ $?
	ranlib $@


