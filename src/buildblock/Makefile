#AIX Makefile for PPhead/buildblock directory
#
# @(#)Makefile	1.24 : 98/11/13
#


# DB 1/2/99 Add WORKSPACE and INCLUDE_DIR directory
WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include


CXX = g++
RM = /bin/rm -f 
# KT 18/12/97 removed Para++ related CFLAGS
#CL030698 Move debug flags to DEBUG
# DB 1/2/99 change CFLAGS
DEBUG=-D_DEBUG
DEST=debug/
CFLAGS = -g $(DEBUG) -I$(INCLUDE_DIR)
#CXX = xlC
#CFLAGS = -g -D_DEBUG -I$(INCLUDE_DIR) -DTEMPLATE_ARG

#KT 27/02/98 use default rules now, .cxx does not work without next line
.SUFFIXES: .cc .o .cxx .a

# KT 04/11/98 enabled built-in sccs
#.SCCS_GET:

.KEEP_STATE:

#KT 16/12 wanted to add interfile...
#CL 060798 Add createdir for create directories of debugged or optimized files versions
all: createdir $(DEST)libbuildblock.a

createdir: 
	-mkdir -p $(DEST)


$(DEST)%.o : %.cxx 
	$(CXX) $(CFLAGS) -o $@ -c $< 

# CL 300998 added PETScanInfo.cxx  file
# KT 22/10/98 added InterfileHeader.o
# KT 04/11/98 added Tensors.o
# KT 06/11/98 added imagedata.o
# KT 22/02/99 added PETDataSymmetries.o PETSinogramOfVolume.o RelatedViewgrams.o
# KT 17/03/99 changed Related... to PETRelated...
$(DEST)libbuildblock.a: $(DEST)sinodata.o $(DEST)convert.o \
  $(DEST)PETScannerInfo.o $(DEST)PETScanInfo.o \
  $(DEST)Tensor1D.o $(DEST)Tensors.o \
  $(DEST)viewdata.o $(DEST)imagedata.o \
  $(DEST)PETDataSymmetries.o \
  $(DEST)PETSinogramOfVolume.o \
  $(DEST)PETRelatedViewgrams.o \
  $(DEST)utilities.o $(DEST)NumericInfo.o \
  ${DEST}KeyParser.o ${DEST}line.o \
  ${DEST}interfile.o ${DEST}InterfileHeader.o \
  ${DEST}bbfilters.o ${DEST}recon_array_functions.o \
  ${DEST}sparsedata.o
	ar r $@ $?
	ranlib $@


#KT 30/07/98 added DEST
clean: 
	$(RM) $(DEST)*.o
	$(RM) $(DEST)*.a

#############################################################

#CL 09/10/98 ADd the header file dependencies for all object files
#I agree that this Makefile become big but it is the only way to update 
#correctly  all the object files

$(DEST)NumericInfo.o: NumericInfo.cxx ../include/NumericInfo.h \
 ../include/pet_common.h

$(DEST)PETScanInfo.o: PETScanInfo.cxx ../include/pet_common.h \
 ../include/PETScanInfo.h ../include/PETScannerInfo.h

$(DEST)PETScannerInfo.o: PETScannerInfo.cxx ../include/pet_common.h \
 ../include/PETScannerInfo.h

# KT 22/10/98 removed higher dimensional dependencies
$(DEST)Tensor1D.o: Tensor1D.cxx ../include/pet_common.h ../include/Tensor1D.h \
 ../include/NumericInfo.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/convert.h 

# KT 04/11/98 new
$(DEST)Tensors.o: Tensors.cxx ../include/pet_common.h \
 ../include/NumericInfo.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/convert.h \
 ../include/Tensor1D.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/Tensor4D.h

$(DEST)convert.o: convert.cxx ../include/pet_common.h ../include/Tensor1D.h \
 ../include/NumericInfo.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/convert.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/Tensor4D.h

# KT 22/10/98 removed
#$(DEST)interfilereader.o: interfilereader.cxx

# KT 22/10/98 moved down to group Interfile stuff
$(DEST)KeyParser.o: KeyParser.cxx ../include/KeyParser.h \
 ../include/NumericInfo.h ../include/pet_common.h ../include/line.h

#CL 19/10/98 ADd this file as I have forgotten to include it
# KT 22/10/98 add InterfileHeader dependency
$(DEST)interfile.o: interfile.cxx ../include/interfile.h \
 ../include/InterfileHeader.h \
 ../include/KeyParser.h ../include/NumericInfo.h \
 ../include/pet_common.h ../include/imagedata.h ../include/Tensor2D.h \
 ../include/Tensorbase.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/Tensor1D.h \
 ../include/Tensor3D.h ../include/Point3D.h ../include/debug.h \
 ../include/utilities.h ../include/sinodata.h \
 ../include/PETScannerInfo.h

# KT 22/10/98 add InterfileHeader 
$(DEST)InterfileHeader.o: InterfileHeader.cxx \
 ../include/interfile.h \
 ../include/InterfileHeader.h \
 ../include/pet_common.h

$(DEST)line.o: line.cxx ../include/line.h

# KT 22/10/98 removed
#$(DEST)pet_string.o: pet_string.cxx ../include/pet_string.h

$(DEST)sinodata.o: sinodata.cxx ../include/pet_common.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h

$(DEST)utilities.o: utilities.cxx ../include/utilities.h \
 ../include/pet_common.h ../include/sinodata.h ../include/Tensor1D.h \
 ../include/NumericInfo.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/Tensor2D.h \
 ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h

$(DEST)viewdata.o: viewdata.cxx ../include/pet_common.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h

#CL 21/10/98 Add the bbfilters.o
$(DEST)bbfilters.o: bbfilters.cxx ../include/pet_common.h \
 ../include/bbfilters.h ../include/PETScannerInfo.h \
 ../include/imagedata.h ../include/Tensor2D.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor1D.h ../include/Tensor3D.h \
 ../include/Point3D.h ../include/debug.h ../include/TensorFunction.h

#MJ 17/11/98 Add the recon_array_functions.o
$(DEST)recon_array_functions.o: recon_array_functions.cxx \
 ../include/pet_common.h ../include/recon_array_functions.h \
 ../include/sinodata.h ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScanInfo.h ../include/PETScannerInfo.h \
 ../include/imagedata.h ../include/Point3D.h ../include/debug.h \
 ../include/TensorFunction.h

#PV 24/11/98 Add the sparsedata.o
$(DEST)sparsedata.o: sparsedata.cxx \
 ../include/sparsedata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScanInfo.h ../include/PETScannerInfo.h \
 ../include/imagedata.h ../include/Point3D.h ../include/debug.h \
 ../include/TensorFunction.h
