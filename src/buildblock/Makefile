#
# $Id$: $Date$
#

include ../Makefile_common

all: $(DEST)libbuildblock.a

clean:
	$(RM) $(DEST)*


SOURCES = \
  Array.cxx convert_array.cxx \
  IndexRange.cxx \
  DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid.cxx \
  ProjData.cxx \
  ProjDataInfo.cxx \
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
  ImageFilter.cxx recon_array_functions.cxx \
  linear_regression.cxx overlap_interpolate.cxx \
  error.cxx warning.cxx  
  



OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


