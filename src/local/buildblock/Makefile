#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libbuildblock.a $(DEST)local_buildblock_registries.o

clean:
	$(RM) $(DEST)*
SOURCES = \
	SeparableLowPassArrayFilter.cxx \
	SeparableLowPassImageFilter.cxx \
	ModifiedInverseAverigingArrayFilter.cxx \
	ModifiedInverseAverigingImageFilter.cxx \
	DAVArrayFilter3D.cxx \
	DAVImageFilter3D.cxx \
	ArrayFilter1DUsingConvolution.cxx \
	fft.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


