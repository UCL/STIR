#AIX Makefile for PPhead/buildblock directory
#
# $Id$: $Date$
#

# AZ 03/10/99 Added
__TEST_OS = $(shell uname)
ifeq ($(__TEST_OS),Linux)
__TEST_LINUX = 1
else
__TEST_LINUX = 0
endif

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include

#AZ 03/02/99 Added

ifeq ($(DEST),debug/)
ifeq ($(__TEST_LINUX),1)
DEBUG = -D_DEBUG -pg
else # not Linux
DEBUG = -D_DEBUG
endif # Linux?
else # not debug/
ifeq ($(DEST),opt/)
ifeq ($(__TEST_LINUX),1)
#-DNDEBUG is needed by system's assert
DEBUG = -O3 -finline-functions -ffast-math -m486 -malign-double -malign-functions=2 -malign-loops=2 -malign-jumps=2 -DCPU=686 -Winline -DNDEBUG
#TODO put this back [incompatible wiht -pg] -fomit-frame-pointer 
# -fforce-mem
else # not Linux
DEBUG = -O3
endif # Linux?
endif # opt/ ?
endif # debug/ ?

CXX = g++
RM = /bin/rm -f 
DEST=debug/
CFLAGS = -g $(DEBUG) -I$(INCLUDE_DIR)

#KT 27/02/98 use default rules now, .cxx does not work without next line
.SUFFIXES: .cc .o .cxx .a

# KT 04/11/98 enabled built-in sccs
#.SCCS_GET:

.KEEP_STATE:

all: createdir $(DEST)libbuildblock.a

createdir: 
	-mkdir -p $(DEST)


$(DEST)%.o : %.cxx 
	$(CXX) $(CFLAGS) -o $@ -c $< 

# KT 10/12/99 added linear_regression.cxx
SOURCES = \
  sinodata.cxx convert.cxx \
  PETScannerInfo.cxx PETScanInfo.cxx \
  Tensor1D.cxx Tensors.cxx \
  viewdata.cxx imagedata.cxx \
  PETDataSymmetries.cxx \
  PETSinogramOfVolume.cxx \
  PETRelatedViewgrams.cxx \
  utilities.cxx NumericInfo.cxx \
  KeyParser.cxx line.cxx \
  interfile.cxx InterfileHeader.cxx \
  bbfilters.cxx recon_array_functions.cxx \
  sparsedata.cxx \
  linear_regression.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

$(DEST)libbuildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@


# KT 18/03/99 new
makedepend:
	ppdep -I ../include $(SOURCES)

#KT 30/07/98 added DEST
clean: 
	$(RM) $(DEST)*.o
	$(RM) $(DEST)*.a

# DO NOT DELETE THIS LINE -- g++dep uses it.
# DO NOT PUT ANYTHING AFTER THIS LINE, IT WILL GO AWAY.

$(DEST)sinodata.o: sinodata.cxx \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/NumericInfo.h \
  ../include/Tensor2D.h \
  ../include/Tensor1D.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/PETSinogram.h \
  ../include/PETViewgram.h
$(DEST)convert.o: convert.cxx \
  ../include/Tensor1D.h \
  ../include/NumericInfo.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/convert.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/Tensor3D.h \
  ../include/Tensor4D.h
$(DEST)PETScannerInfo.o: PETScannerInfo.cxx \
  ../include/PETScannerInfo.h
$(DEST)PETScanInfo.o: PETScanInfo.cxx \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h
$(DEST)Tensor1D.o: Tensor1D.cxx \
  ../include/Tensor1D.h \
  ../include/NumericInfo.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/convert.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/Tensor3D.h \
  ../include/Tensor4D.h
$(DEST)Tensors.o: Tensors.cxx \
  ../include/convert.h \
  ../include/NumericInfo.h \
  ../include/Tensor1D.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/Tensor3D.h \
  ../include/Tensor4D.h
$(DEST)viewdata.o: viewdata.cxx \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/NumericInfo.h \
  ../include/PETSinogram.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor1D.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl
$(DEST)imagedata.o: imagedata.cxx \
  ../include/Tensor2D.h \
  ../include/pet_common.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/NumericInfo.h \
  ../include/Tensor1D.h \
  ../include/Tensor3D.h \
  ../include/Point3D.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/debug.h
$(DEST)PETDataSymmetries.o: PETDataSymmetries.cxx \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl
$(DEST)PETSinogramOfVolume.o: PETSinogramOfVolume.cxx \
  \
  ../include/PETSinogramOfVolume.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/NumericInfo.h \
  ../include/pet_common.h \
  ../include/PETSinogram.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor1D.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl
$(DEST)PETRelatedViewgrams.o: PETRelatedViewgrams.cxx \
  \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/Tensor2D.h \
  ../include/pet_common.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/NumericInfo.h \
  ../include/Tensor1D.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/recon_buildblock/zoom.h \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETSinogram.h \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl \
  ../include/imagedata.h \
  ../include/Point3D.h \
  ../include/debug.h
$(DEST)utilities.o: utilities.cxx \
  ../include/pet_common.h \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/NumericInfo.h \
  ../include/PETSinogram.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor1D.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl \
  ../include/imagedata.h \
  ../include/Point3D.h \
  ../include/debug.h
$(DEST)NumericInfo.o: NumericInfo.cxx \
  ../include/pet_common.h
$(DEST)KeyParser.o: KeyParser.cxx \
  ../include/pet_common.h \
  ../include/line.h
$(DEST)line.o: line.cxx \
  ../include/pet_common.h
$(DEST)interfile.o: interfile.cxx \
  ../include/NumericInfo.h \
  ../include/pet_common.h \
  ../include/KeyParser.h \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/PETSinogram.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor1D.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl \
  ../include/interfile.h \
  ../include/imagedata.h \
  ../include/Point3D.h \
  ../include/debug.h \
  ../include/utilities.h
$(DEST)InterfileHeader.o: InterfileHeader.cxx \
  ../include/NumericInfo.h \
  ../include/pet_common.h \
  ../include/KeyParser.h \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/PETSinogram.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor1D.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl
$(DEST)bbfilters.o: bbfilters.cxx \
  ../include/bbfilters.h \
  ../include/PETScannerInfo.h \
  ../include/imagedata.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/NumericInfo.h \
  ../include/Tensor1D.h \
  ../include/Tensor3D.h \
  ../include/Point3D.h \
  ../include/PETScanInfo.h \
  ../include/debug.h \
  ../include/TensorFunction.h \
  ../include/utilities.h \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETSinogram.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/PETSinogramOfVolume.inl
$(DEST)recon_array_functions.o: recon_array_functions.cxx \
  \
  ../include/pet_common.h \
  ../include/recon_array_functions.h \
  ../include/sinodata.h \
  ../include/PETSinogramOfVolume.h \
  ../include/PETScanInfo.h \
  ../include/PETScannerInfo.h \
  ../include/NumericInfo.h \
  ../include/PETSinogram.h \
  ../include/Tensor2D.h \
  ../include/Tensorbase.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor1D.h \
  ../include/PETRelatedViewgrams.h \
  ../include/PETViewgram.h \
  ../include/PETDataSymmetries.h \
  ../include/ViewSegmentNumbers.h \
  ../include/PETDataSymmetries.inl \
  ../include/PETRelatedViewgrams.inl \
  ../include/PETSegment.h \
  ../include/Tensor3D.h \
  ../include/PETSinogramOfVolume.inl \
  ../include/imagedata.h \
  ../include/Point3D.h \
  ../include/debug.h \
  ../include/TensorFunction.h
$(DEST)sparsedata.o: sparsedata.cxx \
  ../include/Tensor1D.h \
  ../include/pet_common.h \
  ../include/NumericInfo.h \
  ../include/NumericVectorWithOffset.h \
  ../include/VectorWithOffset.h \
  ../include/Tensor3D.h \
  ../include/Tensorbase.h \
  ../include/Tensor2D.h \
  ../include/Point3D.h \
  ../include/PETScannerInfo.h

# IF YOU PUT ANYTHING HERE IT WILL GO AWAY
