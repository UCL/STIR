#
# AIX Makefile for PPhead/test directory
#
# $Id$: $Date$
#
# 
WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include


#These libraries will be used as dependencies
LIB_RECON_BUILDINGBLOCK= \
   ${WORKSPACE}/recon_buildblock/${DEST}librecon_buildblock.a
LIB_BUILDINGBLOCK= \
   ${WORKSPACE}/buildblock/${DEST}libbuildblock.a
LIB_DISPLAY= \
   ${WORKSPACE}/display/${DEST}libdisplay.a




AR = ar
RANLIB= ranlib
DEBUG= -D_DEBUG
DEST=debug/
CXX= g++ 
PXCXX = px g++


#KT 22/05/98 use INCLUDE_DIR and DEBUG and more warnings
CFLAGS= -g $(DEBUG) -Decho -I$(INCLUDE_DIR) -Wconversion -Wstrict-prototypes 
#LDFLAGS= -Iinclude 

.SUFFIXES: .cc .o .cxx .a

${DEST}%.o : %.cxx
#.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 


${DEST}% : ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
#.o:
	$(CXX) $(CFLAGS) -o $@ $< \
	     ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}


RM = /bin/rm -f 

LIBS1 = -lm

#KT 22/05/98 use GNU make specific syntax to differentiate between systems (sorry)
ifdef CYGROOT
#Cygwin32 doesn't have curses
LIBS = $(LIBS1) -L/usr/X11R6.3/lib -lX11 
else
LIBS = $(LIBS1)  -lX11 -lcurses
endif
#LIBS_SOL = -L/usr/openwin/lib -L../../../LIBECAT/SOL2  -lm -lecat -lnsl -lsocket

# From here specific to this Makefile


all: createdir libs nopara
 
createdir:
	-mkdir -p $(DEST)

libs: $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)

#CL 12/11/98 Remove $(DEST)sinodata.cxx  as it is a very old version and rename sinodata.cxx
nopara: $(DEST)tensor $(DEST)display2 \
	$(DEST)VectorWithOffset $(DEST)testconvert

#$(DEST)sinodata

$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE);

# KT 09/10/98 changed spelling mistake BUILD -> BUILDING
$(LIB_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); 

FORCE:

clean:
	rm $(DEST)tensor $(DEST)display $(DEST)sinodata  \
	$(DEST)VectorWithOffsetc $(DEST)testconvert 

#CL 09/10/98 ADd the header file dependencies for all object files
#I agree that this Makefile become big but it is the only way to update 
#correctly  all the object files

$(DEST)Timer.o: Timer.cxx ../include/Timer.h

$(DEST)VectorWithOffset.o: VectorWithOffset.cxx ../include/VectorWithOffset.h \
 ../include/pet_common.h

$(DEST)createsino.o: createsino.cxx ../include/Tensor3D.h \
 ../include/pet_common.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor2D.h ../include/Tensor1D.h

$(DEST)createsino2.o: createsino2.cxx ../include/Tensor3D.h \
 ../include/pet_common.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor2D.h ../include/Tensor1D.h

$(DEST)display2.o: display2.cxx ../include/pet_common.h \
 ../include/VectorWithOffset.h ../include/Tensor2D.h \
 ../include/Tensorbase.h ../include/NumericVectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor1D.h ../include/Tensor3D.h \
 ../include/display.h ../include/sinodata.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h

$(DEST)sinodata2.o: sinodata2.cxx ../include/pet_common.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h \
 ../include/display.h

$(DEST)tensor.o: tensor.cxx ../include/Tensor4D.h ../include/pet_common.h \
 ../include/Tensorbase.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/NumericInfo.h \
 ../include/Tensor3D.h ../include/Tensor2D.h ../include/Tensor1D.h \
 ../include/TensorFunction.h

$(DEST)testconvert.o: testconvert.cxx ../include/convert.h \
 ../include/NumericInfo.h ../include/pet_common.h \
 ../include/Tensor1D.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/Tensor2D.h \
 ../include/Tensorbase.h ../include/Tensor3D.h ../include/Tensor4D.h

