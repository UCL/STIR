#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libbuildblock.a $(DEST)local_buildblock_registries.o

clean:
	$(RM) $(DEST)*

SOURCES = \
	DAVArrayFilter3D.cxx \
	DAVImageFilter3D.cxx \
	fft.cxx \
	SSRB.cxx \
	ArrayFilter3DUsingConvolution.cxx \
	ArrayFilter2DUsingConvolution.cxx \
	ModifiedInverseAveragingImageFilterAll.cxx \
	fwd_and_bck_manipulation_for_SAF.cxx \
	ML_norm.cxx \
	SeparableLowPassArrayFilter.cxx \
	SeparableLowPassImageFilter.cxx \
	ModifiedInverseAverigingArrayFilter.cxx \
	ModifiedInverseAverigingImageFilter.cxx \
	SeparableGaussianImageFilter.cxx \
	SeparableGaussianArrayFilter.cxx \
	NonseparableSpatiallyVaryingFilters.cxx \
	NonseparableSpatiallyVaryingFilters3D.cxx \
	local_helping_functions.cxx


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

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


