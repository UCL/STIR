#
# $Id$: $Date$
#

include ../Makefile_common

all: $(DEST)libbuildblock.a

clean:
	$(RM) $(DEST)*

SOURCES = \
  PETerror.cxx \
  sinodata.cxx convert.cxx \
  PETScannerInfo.cxx PETScanInfo.cxx \
  Tensor1D.cxx Tensors.cxx \
  viewdata.cxx imagedata.cxx \
  PETDataSymmetries.cxx \
  PETSinogramOfVolume.cxx \
  PETRelatedViewgrams.cxx \
  utilities.cxx NumericType.cxx ByteOrder.cxx \
  KeyParser.cxx line.cxx \
  interfile.cxx InterfileHeader.cxx \
  bbfilters.cxx recon_array_functions.cxx \
  sparsedata.cxx \
  linear_regression.cxx overlap_interpolate.cxx \
  zoom.cxx mash_views.cxx 

SOURCES_NEW=  Array.cxx convert_array.cxx \
  VoxelsOnCartesianGrid.cxx \
  IndexRange.cxx error.cxx warning.cxx


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

.PRECIOUS: $(DEST)*.o $(DEST)*.a

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


