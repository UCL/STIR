#
# $Id$ %E
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include
RANLIB= ranlib

#These libraries will be used as dependencies
LIB_BUILDINGBLOCK= \
   ${WORKSPACE}/buildblock/${DEST}libbuildblock.a
LIB_DISPLAY= \
   ${WORKSPACE}/display/${DEST}libdisplay.a



DEBUG= -D_DEBUG
DEST=debug/
CXX= g++ 


#KT 22/05/98 use INCLUDE_DIR and DEBUG and more warnings
CFLAGS= -g $(DEBUG) -Decho -I$(INCLUDE_DIR) -Wconversion -Wstrict-prototypes 
#LDFLAGS= -Iinclude 

.SCCS_GET:

.SUFFIXES: .cc .o .cxx .a


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

# KT 09/10/98 use OBJS,EXES to prevent repetition, add displayIF
#CL 10/10/98 Add readsino
SOURCES = readsino.cxx displayPSOV.cxx swap_byteorder.cxx ecat6bysegm.cxx ecat6bysegmspan.cxx conv2cti.cxx #displayIF.cxx vox.cxx CL 20/10/98 Disable vox as Interfile library not compiled properly

OBJS=$(SOURCES:%.cxx=$(DEST)%.o)
EXES=$(SOURCES:%.cxx=$(DEST)%)

#CL 12/10/98 ADd the utility to convert bitmap en CTI format
SOURCES_CTI = conv2cti.cxx saveImageCTI.cxx reconstruction.cxx voxelphantom.cxx cti_utils.cxx
OBJS_CTI= $(SOURCES_CTI:%.cxx=$(DEST)%.o)

# KT 09/10/98 changed to OBJS
.PRECIOUS: $(OBJS)



all: createdir libs $(EXES) 
 
createdir:
	-mkdir -p $(DEST)

$(DEST)conv2cti:  $(OBJS_CTI) 
	$(CXX) $(CFLAGS)  -o $@ $(OBJS_CTI)

#CL 09/03/98 NEW SECTION for conversion of an ECAT6 file to ECAT6 file organized by segment (0, +1, -1, +2, -2,...)

$(DEST)ecat6bysegm:  ecat6bysegm.cxx $(DEST)libcti.a 
	$(CXX) $(CFLAGS)  -o $@  $< -L$(DEST) -lcti -lm 

$(DEST)ecat6bysegmspan: ecat6bysegmspan.cxx $(DEST)libcti.a 
	$(CXX) $(CFLAGS)  -o $@ ecat6bysegmspan.cxx -L$(DEST)  -lcti -lm 

#CTI Libraries $(LIB_CTI) have been built
$(DEST)libcti.a: $(DEST)cti_utils.o $(DEST)reconstruction.o $(DEST)fft.o
	$(AR) r $@ $?
	$(RANLIB) $@

$(DEST)readsino: readsino.cxx  libs 
	$(CXX) $(CFLAGS) -o $@ $< -I../display \
	    ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}


libs: $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)


$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE);

# KT 09/10/98 changed spelling mistake BUILD -> BUILDING
$(LIB_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); 

#CL 12/10/98 Add the header files dependencies

$(DEST)displayIF.o: displayIF.cxx ../include/pet_common.h \
 ../include/imagedata.h ../include/Tensor2D.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor1D.h ../include/Tensor3D.h \
 ../include/Point3D.h ../include/debug.h ../include/interfile.h \
 ../include/KeyParser.h ../include/display.h

$(DEST)displayPSOV.o: displayPSOV.cxx ../include/pet_common.h \
 ../include/sinodata.h ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/display.h \
 ../include/utilities.h ../include/imagedata.h ../include/Point3D.h \
 ../include/debug.h ../display/gen.h ../include/pet_common.h

$(DEST)readsino.o: readsino.cxx ../include/pet_common.h \
 ../include/PETScannerInfo.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/imagedata.h ../include/Point3D.h ../include/debug.h \
 ../include/convert.h ../include/Tensor4D.h ../include/display.h \
 ../display/gen.h

$(DEST)swap_byteorder.o: swap_byteorder.cxx ../include/pet_common.h

$(DEST)/vox.o: vox.cxx ../include/pet_common.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/imagedata.h \
 ../include/Point3D.h ../include/debug.h ../include/TensorFunction.h \
 ../include/display.h ../include/recon_buildblock/bckproj.h \
 ../include/Tensor4D.h ../include/recon_buildblock/fwdproj.h \
 ../include/recon_buildblock/zoom.h


$(DEST)conv2cti.o: conv2cti.cxx ../include/CTI/camera.h \
 ../include/CTI/reconstruction.h ../include/CTI/rcn_config.h \
 ../include/CTI/rcn_types.h ../include/CTI/cti_types.h \
 ../include/CTI/pet_utils.h ../include/CTI/cti_utils.h \
 ../include/CTI/voxelphantom.h

$(DEST)saveImageCTI.o: saveImageCTI.cxx ../include/CTI/pet_utils.h \
 ../include/CTI/cti_utils.h ../include/CTI/rcn_config.h \
 ../include/CTI/camera.h ../include/CTI/cti_types.h \
 ../include/CTI/rcn_types.h ../include/CTI/reconstruction.h

$(DEST)reconstruction.o: reconstruction.cxx ../include/CTI/reconstruction.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/rcn_types.h ../include/CTI/pet_utils.h \
 ../include/CTI/cti_utils.h ../include/CTI/cti_types.h

$(DEST)voxelphantom.o: voxelphantom.cxx ../include/CTI/voxelphantom.h
cti_utils.o: cti_utils.cxx ../include/CTI/cti_utils.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/cti_types.h ../include/CTI/rcn_types.h

ecat6bysegm.o: ecat6bysegm.cxx ../include/CTI/rcn_types.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/reconstruction.h ../include/CTI/cti_utils.h \
 ../include/CTI/cti_types.h

ecat6bysegmspan.o: ecat6bysegmspan.cxx ../include/CTI/rcn_types.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/reconstruction.h ../include/CTI/cti_utils.h \
 ../include/CTI/cti_types.h

$(DEST)%.o : %.cxx
	$(CXX) $(CFLAGS) -o $@ -c $< 

$(DEST)% : $(DEST)%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS) -o $@ $< \
	    ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}



FORCE:


clean: 
	$(RM) $(DEST)*.o $(EXES)
