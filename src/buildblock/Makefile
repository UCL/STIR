#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libbuildblock.a $(DEST)buildblock_registries.o

clean:
	$(RM) $(DEST)*.[oadP]

# KT 08/02/2002
# added ProjDataInMemory.cxx
# KT 07/05/2001 added ArrayFilter1DUsingConvolution.cxx and SeparableConvolutionImageFilter.cxx
SOURCES = \
  Array.cxx convert_array.cxx \
  IndexRange.cxx \
  ProjData.cxx \
  ProjDataInfo.cxx \
  ProjDataInfoCylindrical.cxx \
  ProjDataInfoCylindricalArcCorr.cxx \
  ProjDataInfoCylindricalNoArcCorr.cxx \
  ProjDataFromStream.cxx \
  ProjDataGEAdvance.cxx \
  ProjDataInMemory.cxx \
  ProjDataInterfile.cxx \
  Scanner.cxx \
  SegmentBySinogram.cxx \
  SegmentByView.cxx \
  Viewgram.cxx \
  Sinogram.cxx \
  RelatedViewgrams.cxx \
  DiscretisedDensity.cxx \
  VoxelsOnCartesianGrid.cxx \
  utilities.cxx \
  interfile_keyword_functions.cxx \
  zoom.cxx \
  NumericType.cxx ByteOrder.cxx \
  KeyParser.cxx  \
  recon_array_functions.cxx \
  linear_regression.cxx overlap_interpolate.cxx \
  error.cxx warning.cxx  \
  DataSymmetriesForViewSegmentNumbers.cxx \
	ParsingObject.cxx \
	ArrayFilter1DUsingConvolutionSymmetricKernel.cxx \
	SeparableArrayFunctionObject.cxx \
	SeparableMetzArrayFilter.cxx \
	MedianArrayFilter3D.cxx \
	MedianImageFilter3D.cxx \
	SeparableCartesianMetzImageFilter.cxx \
	ThresholdMinToSmallPositiveValueImageProcessor.cxx \
	ChainedImageProcessor.cxx \
	ArrayFilter1DUsingConvolution.cxx \
	SeparableConvolutionImageFilter.cxx \
	SSRB.cxx
 




OBJS = $(SOURCES:%.cxx=$(DEST)%.o)
#LOBJS = $(DEST)libbuildblock.a($(OBJS))

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
  include $(DEST)buildblock_registries.d
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
  -include $(DEST)buildblock_registries.P
endif
endif


.PRECIOUS:  $(DEST)*.a $(DEST)*.o



$(DEST)libbuildblock.a: $(OBJS)
	$(AR) $(ARFLAGS)  $@ $?
	ranlib $@


