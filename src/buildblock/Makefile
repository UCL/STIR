#
# $Id$
#


CXX = g++
RM = /bin/rm -f 
# KT 18/12/97 removed Para++ related CFLAGS
#CL030698 Move debug flags to DEBUG
DEBUG=-D_DEBUG
DEST=debug/
CFLAGS = -g $(DEBUG) -I${HOME}/PPhead/include
#CXX = xlC
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEMPLATE_ARG

#KT 27/02/98 use default rules now, .cxx does not work without next line
.SUFFIXES: .cc .o .cxx .a

.KEEP_STATE:

#KT 16/12 wanted to add interfile...
#CL 060798 Add createdir for create directories of debugged or optimized files versions
all: createdir ${DEST}libbuildblock.a #${DEST}libinterfile.a

createdir: 
	-mkdir -p $(DEST)


${DEST}%.o : %.cxx 
	$(CXX) $(CFLAGS) -o $@ -c $< 

#CL 09/10/98 ADd the header file dependencies for all object files
#I agree that this Makefile become big but it is the only way to update 
#correctly  all the object files

$(DEST)KeyParser.o: KeyParser.cxx ../include/KeyParser.h \
 ../include/NumericInfo.h ../include/pet_common.h ../include/line.h

$(DEST)NumericInfo.o: NumericInfo.cxx ../include/NumericInfo.h \
 ../include/pet_common.h

$(DEST)PETScanInfo.o: PETScanInfo.cxx ../include/pet_common.h \
 ../include/PETScanInfo.h ../include/PETScannerInfo.h

$(DEST)PETScannerInfo.o: PETScannerInfo.cxx ../include/pet_common.h \
 ../include/PETScannerInfo.h

$(DEST)Tensor1D.o: Tensor1D.cxx ../include/pet_common.h ../include/Tensor1D.h \
 ../include/NumericInfo.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/convert.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/Tensor4D.h

$(DEST)convert.o: convert.cxx ../include/pet_common.h ../include/Tensor1D.h \
 ../include/NumericInfo.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/convert.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/Tensor4D.h

$(DEST)interfilereader.o: interfilereader.cxx

$(DEST)line.o: line.cxx ../include/line.h

$(DEST)pet_string.o: pet_string.cxx ../include/pet_string.h

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

# KT 30/07/98 added utilities
# KT 03/09/98 added NumericInfo
#CL 300998 added PETScanInfo.cxx  file
${DEST}libbuildblock.a: ${DEST}sinodata.o ${DEST}convert.o \
   ${DEST}PETScannerInfo.o ${DEST}PETScanInfo.o ${DEST}Tensor1D.o ${DEST}viewdata.o \
   $(DEST)utilities.o $(DEST)NumericInfo.o
	ar r $@ $?
	ranlib $@

# KT 03/09/98 removed pet_string
${DEST}libinterfile.a: ${DEST}KeyParser.o ${DEST}line.o 

#KT 30/07/98 added DEST
clean: 
	$(RM) $(DEST)*.o
	$(RM) $(DEST)*.a
