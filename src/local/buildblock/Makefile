#
# $Id$
#

include $(WORKSPACE)/Makefile_common


all: $(DEST)libbuildblock.a $(DEST)local_buildblock_registries.o

clean:
	$(RM) $(DEST)*
SOURCES = \
	DAVArrayFilter3D.cxx \
	DAVImageFilter3D.cxx \


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


