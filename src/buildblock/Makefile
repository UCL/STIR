#
# $Id$
#

include ../Makefile_common


all: $(DEST)libbuildblock.a $(DEST)buildblock_registries.o

clean:
	$(RM) $(DEST)*

# KT 25/10/2000
# removed   DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid.cxx 
# added   ProjDataInfoCylindrical.cxx 
# KT 25/04/2001 added new filter stuff, removed ImageFilter.cxx
SOURCES = \
  Array.cxx convert_array.cxx \
  IndexRange.cxx \
  ProjData.cxx \
  ProjDataInfo.cxx \
  ProjDataInfoCylindrical.cxx \
  ProjDataFromStream.cxx \
  ProjDataGEAdvance.cxx \
  Scanner.cxx \
  SegmentBySinogram.cxx \
  SegmentByView.cxx \
  Viewgram.cxx \
  Sinogram.cxx \
  RelatedViewgrams.cxx \
  DiscretisedDensity.cxx \
  VoxelsOnCartesianGrid.cxx \
  utilities.cxx \
  interfile.cxx InterfileHeader.cxx \
  zoom.cxx \
  NumericType.cxx ByteOrder.cxx \
  KeyParser.cxx line.cxx \
  recon_array_functions.cxx \
  linear_regression.cxx overlap_interpolate.cxx \
  error.cxx warning.cxx  \
	ParsingObject.cxx \
	ArrayFilter1DUsingConvolutionSymmetricKernel.cxx \
	SeparableArrayFunctionObject.cxx \
	SeparableMetzArrayFilter.cxx \
	MedianArrayFilter3D.cxx \
	MedianImageFilter3D.cxx \
	SeparableCartesianMetzImageFilter.cxx 
 




OBJS = $(SOURCES:%.cxx=$(DEST)%.o)
LOBJS = $(DEST)libbuildblock.a($(OBJS))

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


.PRECIOUS:  $(DEST)*.a $(DEST)*.o


test:
	 echo "$(LOBJS)"

$(DEST)libbuildblock.a: $(LOBJS)
	ranlib $@
#	ar r $@ $?


