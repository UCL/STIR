#
# $Id$
#

include ../../Makefile_common


all: $(DEST)libbuildblock.a $(DEST)registrystuff.o

clean:
	$(RM) $(DEST)*
SOURCES = \
	DAVArrayFilter3D.cxx \
	DAVImageFilter3D.cxx \

 

$(DEST)registrystuff.o: registrystuff.cxx 



OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


