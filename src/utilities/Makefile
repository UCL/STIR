#
# $Id$
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include
RANLIB= ranlib

#These libraries will be used as dependencies
# KT 09/02/98 added reconbuildblock
LIB_RECON_BUILDINGBLOCK= \
   ${WORKSPACE}/recon_buildblock/${DEST}librecon_buildblock.a
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

#KT 12/12/99 removed CYGROOT stuff
LIBS = $(LIBS1) -L/usr/X11/lib -lX11 -lcurses
#LIBS_SOL = -L/usr/openwin/lib -L../../../LIBECAT/SOL2  -lm -lecat -lnsl -lsocket


# KT 09/02/98 added do_linear_regression
SOURCES = displayPSOV.cxx swap_byteorder.cxx ecat6bysegm.cxx \
 ecat6bysegmspan.cxx conv2cti.cxx vox.cxx correctPSOV.cxx \
 postfilter.cxx binmath.cxx do_linear_regression

OBJS=$(SOURCES:%.cxx=$(DEST)%.o)
EXES=$(SOURCES:%.cxx=$(DEST)%)

#CL 12/10/98 ADd the utility to convert bitmap en CTI format
SOURCES_CTI = conv2cti.cxx saveImageCTI.cxx recon_tools.cxx voxelphantom.cxx cti_utils.cxx
OBJS_CTI= $(SOURCES_CTI:%.cxx=$(DEST)%.o)

# KT 09/10/98 changed to OBJS
.PRECIOUS: $(OBJS)



all: createdir libs $(EXES) 
 
createdir:
	-mkdir -p $(DEST)

$(DEST)conv2cti:  $(OBJS_CTI) 
	$(CXX) $(CFLAGS)  -o $@ $(OBJS_CTI)

#CL 09/03/98 NEW SECTION for conversion of an ECAT6 file to ECAT6 file organized by segment (0, +1, -1, +2, -2,...)

# KT 27/11/98 added buildblock for PETScannerInfo.o
$(DEST)ecat6bysegm:  ecat6bysegm.cxx $(DEST)libcti.a  ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@  $< -L$(DEST) -lcti -lm  ${LIB_BUILDINGBLOCK}

$(DEST)ecat6bysegmspan: ecat6bysegmspan.cxx $(DEST)libcti.a 
	$(CXX) $(CFLAGS)  -o $@ ecat6bysegmspan.cxx -L$(DEST)  -lcti -lm 

#CTI Libraries $(LIB_CTI) have been built
$(DEST)libcti.a: $(DEST)cti_utils.o $(DEST)recon_tools.o $(DEST)fft.o
	$(AR) r $@ $?
	$(RANLIB) $@

$(DEST)readsino: readsino.cxx  libs 
	$(CXX) $(CFLAGS) -o $@ $< -I../display \
	    ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}

# KT 09/02/98 added correctPSOV
$(DEST)correctPSOV:  correctPSOV.cxx ${LIB_RECON_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@  $< -L$(DEST)  -lm \
	${LIB_BUILDINGBLOCK} ${LIB_RECON_BUILDINGBLOCK}

# KT 09/02/98 added reconbuildblock
libs: $(LIB_RECON_BUILDINGBLOCK) $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)


$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE);

# KT 09/10/98 changed spelling mistake BUILD -> BUILDING
$(LIB_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); 

# KT 09/02/98 added reconbuildblock
$(LIB_RECON_BUILDINGBLOCK) : FORCE
	cd $(WORKSPACE)/recon_buildblock; $(MAKE) 


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

# KT 09/02/98 removed / after (DEST)
$(DEST)vox.o: vox.cxx ../include/pet_common.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/imagedata.h \
 ../include/Point3D.h ../include/debug.h ../include/TensorFunction.h \
 ../include/display.h ../include/recon_buildblock/bckproj.h \
 ../include/Tensor4D.h ../include/recon_buildblock/fwdproj.h \
 ../include/recon_buildblock/zoom.h

$(DEST)binmath.o: binmath.cxx ../include/pet_common.h ../include/imagedata.h \
 ../include/Tensor2D.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor1D.h ../include/Tensor3D.h \
 ../include/Point3D.h ../include/PETScanInfo.h \
 ../include/PETScannerInfo.h ../include/debug.h \
 ../include/TensorFunction.h ../include/recon_array_functions.h \
 ../include/sinodata.h ../include/PETSinogramOfVolume.h \
 ../include/PETSinogram.h ../include/PETRelatedViewgrams.h \
 ../include/PETViewgram.h ../include/PETDataSymmetries.h \
 ../include/ViewSegmentNumbers.h ../include/PETDataSymmetries.inl \
 ../include/PETRelatedViewgrams.inl ../include/PETSegment.h \
 ../include/PETSinogramOfVolume.inl ../include/display.h \
 ../include/recon_buildblock/bckproj.h ../include/Tensor4D.h \
 ../include/recon_buildblock/fwdproj.h ../include/interfile.h \
 ../include/KeyParser.h ../include/utilities.h \
 ../include/utilities.inl



$(DEST)conv2cti.o: conv2cti.cxx ../include/CTI/camera.h \
 ../include/CTI/recon_tools.h ../include/CTI/rcn_config.h \
 ../include/CTI/rcn_types.h ../include/CTI/cti_types.h \
 ../include/CTI/pet_utils.h ../include/CTI/cti_utils.h \
 ../include/CTI/voxelphantom.h

$(DEST)saveImageCTI.o: saveImageCTI.cxx ../include/CTI/pet_utils.h \
 ../include/CTI/cti_utils.h ../include/CTI/rcn_config.h \
 ../include/CTI/camera.h ../include/CTI/cti_types.h \
 ../include/CTI/rcn_types.h ../include/CTI/recon_tools.h

$(DEST)recon_tools.o: recon_tools.cxx ../include/CTI/recon_tools.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/rcn_types.h ../include/CTI/pet_utils.h \
 ../include/CTI/cti_utils.h ../include/CTI/cti_types.h

$(DEST)voxelphantom.o: voxelphantom.cxx ../include/CTI/voxelphantom.h
cti_utils.o: cti_utils.cxx ../include/CTI/cti_utils.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/cti_types.h ../include/CTI/rcn_types.h

ecat6bysegm.o: ecat6bysegm.cxx ../include/CTI/rcn_types.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/recon_tools.h ../include/CTI/cti_utils.h \
 ../include/CTI/cti_types.h

ecat6bysegmspan.o: ecat6bysegmspan.cxx ../include/CTI/rcn_types.h \
 ../include/CTI/rcn_config.h ../include/CTI/camera.h \
 ../include/CTI/recon_tools.h ../include/CTI/cti_utils.h \
 ../include/CTI/cti_types.h


$(DEST)postfilter.o: postfilter.cxx ../include/pet_common.h \
 ../include/imagedata.h ../include/Tensor2D.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor1D.h ../include/Tensor3D.h \
 ../include/Point3D.h ../include/PETScanInfo.h \
 ../include/PETScannerInfo.h ../include/debug.h \
 ../include/TensorFunction.h ../include/display.h \
 ../include/sinodata.h ../include/interfile.h ../include/KeyParser.h \
 ../include/utilities.h ../include/recon_array_functions.h \
 ../include/bbfilters.h

$(DEST)%.o : %.cxx
	$(CXX) $(CFLAGS) -o $@ -c $< 

$(DEST)% : $(DEST)%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS) -o $@ $< \
	    ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}



FORCE:


clean: 
	$(RM) $(DEST)*.o $(EXES)
