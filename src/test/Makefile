# @(#)Makefile	1.6 : 97/12/16
.KEEP_STATE:
AR = ar
RANLIB= ranlib
CXX= g++ 
SYS= 

#KT 28/11 some new defines
WORKSPACE = ${HOME}/PPhead
INCLUDE = ${WORKSPACE}/include
DISPLAYLIB = ${WORKSPACE}/display/$(SYS)/libdisplay.a
BUILDBLOCKLIB = ${WORKSPACE}/buildblock/$(SYS)/libbuildblock.a
PARAPLUSLIB = ${WORKSPACE}/paraplus/$(SYS)/libparaplus.a
PXLINK = px g++ -e preMain -I/epx/epx/include -L/epx/epx/lib
PXLIB = -lpx -lparix -lVT -lhslink -lpthreads
LIB_CTI = libcti_utils.a libreconstruction.a libfft.a  

INTERFILELIB = ${WORKSPACE}/buildblock/libinterfile.a
#KT 22/01/98 added this variable to differentiate between programs who use display
# the 2nd -L option is for Solaris
DISPLAYSWITCHES =  -L${WORKSPACE}/display  \
          -L/usr/openwin/lib -ldisplay -lX11 -lcurses

#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
CFLAGS = -g -D_DEBUG -I${INCLUDE} -I/usr/openwin/include -I./libs_cti/include 

LDFLAGS = -g -L${WORKSPACE}/display/$(SYS)  -L${WORKSPACE}/buildblock/$(SYS) \
	  -L${WORKSPACE}/paraplus/$(SYS) \
          -L/usr/openwin/lib  \
          -lbuildblock -ldisplay -lX11 -lcurses
#-lparaplus


.PRECIOUS: $(LIB_CTI) libcti.a


#KT GNU make would allow $+ in the rule below to include all dependents, 
#but Solaris make doesn't...
#CL 18/10/98 Remove ecat6bysegm as it has been moved to utilities
#directory
all: tensor display sinodata createsino VectorWithOffset convert pplusdummy pplustiny #ecat6bysegm

#KT added .h dependencies
tensor: tensor.cxx ${INCLUDE}/Tensor1D.h ${INCLUDE}/Tensor2D.h \
	${INCLUDE}/Tensor3D.h ${INCLUDE}/Tensor4D.h \
	${INCLUDE}/Tensorbase.h ${INCLUDE}/VectorWithOffset.h
	$(CXX) $(CFLAGS) -o $@ $< ${LDFLAGS}

#KT The -I${WORKSPACE}/display flag is only necessary because I used gen.h
#(part of my old library). 
display: display.cxx ${DISPLAYLIB} 
	$(CXX) $(CFLAGS) -I${WORKSPACE}/display -o $@ $< $(LDFLAGS) 

sinodata: sinodata.cxx ${DISPLAYLIB} ${BUILDBLOCKLIB} 
	$(CXX) $(CFLAGS) -o $@ $<  $(LDFLAGS)


createsino: createsino.cxx ${DISPLAYLIB} ${BUILDBLOCKLIB} 
	$(CXX) $(CFLAGS) -o $@ $<  $(LDFLAGS)

# KT 2/03/98 added
#KT The -I${WORKSPACE}/display flag is only necessary because I used gen.h
#(part of my old library). 
displayPSOV: displayPSOV.cxx ${DISPLAYLIB} ${BUILDBLOCKLIB} 
	$(CXX) $(CFLAGS) -o $@  -I${WORKSPACE}/display $<  $(LDFLAGS) 
# ${DISPLAYSWITCHES}

VectorWithOffset: VectorWithOffset.cxx ${INCLUDE}/VectorWithOffset.h

convert: convert.cxx ${BUILDBLOCKLIB}
	$(CXX) $(CFLAGS) -o $@ $< 

testconvert: testconvert.cxx ${BUILDBLOCKLIB}
 
testinterfilereader: testinterfilereader.cxx ${INTERFILEFLIB}
	$(CXX) $(CFLAGS) -o $@ $<  $(LDFLAGS) ${INTERFILELIB}

pplusdummy: pplusdummy.cxx ${PARAPLUSLIB}
	$(CXX) -c $(CFLAGS)  $< 
	$(PXLINK) -o $@ $@.o $(LDFLAGS) $(PXLIB)

pplustiny: pplustiny.cxx ${PARAPLUSLIB}
	$(CXX) -c $(CFLAGS)  $@ $< 
	$(PXLINK) -o $@ $@.o $(LDFLAGS) $(PXLIB)

#CL 090398 NEW SECTION for conversion of an ECAT6 file to ECAT6 file organized by segment (0, +1, -1, +2, -2,...)
ecat6bysegm:  ecat6bysegm.cxx cti 
	$(CXX) $(CFLAGS)  -o $@  $< -L./ -lcti -lm 

ecat6bysegmspan: cti 
	$(CXX) $(CFLAGS)  -o $@ ecat6bysegmspan.cxx -L./  -lcti -lm 


#	 CTI Libraries $(LIB_CTI) have been built
cti : libcti.a #$(LIB_CTI)

libcti.a: libs_cti/cti_utils.o libs_cti/reconstruction.o libs_cti/fft.o
	$(AR) rcv $@ $<
	$(RANLIB) $@


libs_cti/%.o: libs_cti/%.c
	$(CXX) $(DEBUG) $(CFLAGS) -I./include -o $@ -c $<

ParaEPXtest: ParaEPXtest.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)


%.o:%.cxx
	$(CXX) $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX) $(CFLAGS) -o $@ -c $< 
%:%.cxx
	$(CXX) $(CFLAGS) -o $@ $< ${LDFLAGS}


#CL 09/10/98 ADd the header file dependencies for all object files
#I agree that this Makefile become big but it is the only way to update 
#correctly  all the object files

Timer.o: Timer.cxx ../include/Timer.h

VectorWithOffset.o: VectorWithOffset.cxx ../include/VectorWithOffset.h \
 ../include/pet_common.h

createsino.o: createsino.cxx ../include/Tensor3D.h \
 ../include/pet_common.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor2D.h ../include/Tensor1D.h

createsino2.o: createsino2.cxx ../include/Tensor3D.h \
 ../include/pet_common.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor2D.h ../include/Tensor1D.h

display.o: display.cxx ../include/pet_common.h \
 ../include/VectorWithOffset.h ../include/Tensor2D.h \
 ../include/Tensorbase.h ../include/NumericVectorWithOffset.h \
 ../include/NumericInfo.h ../include/Tensor1D.h ../include/Tensor3D.h \
 ../include/display.h ../include/sinodata.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h

ecat6bysegm.o: ecat6bysegm.cxx libs_cti/include/rcn_types.h \
 libs_cti/include/rcn_config.h libs_cti/include/camera.h \
 libs_cti/include/reconstruction.h libs_cti/include/cti_utils.h \
 libs_cti/include/cti_types.h

ecat6bysegmspan.o: ecat6bysegmspan.cxx libs_cti/include/rcn_types.h \
 libs_cti/include/rcn_config.h libs_cti/include/camera.h \
 libs_cti/include/reconstruction.h libs_cti/include/cti_utils.h \
 libs_cti/include/cti_types.h

sinodata.o: sinodata.cxx ../include/pet_common.h ../include/sinodata.h \
 ../include/Tensor1D.h ../include/NumericInfo.h \
 ../include/NumericVectorWithOffset.h ../include/VectorWithOffset.h \
 ../include/Tensor2D.h ../include/Tensorbase.h ../include/Tensor3D.h \
 ../include/PETScannerInfo.h ../include/PETScanInfo.h \
 ../include/display.h

tensor.o: tensor.cxx ../include/Tensor4D.h ../include/pet_common.h \
 ../include/Tensorbase.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/NumericInfo.h \
 ../include/Tensor3D.h ../include/Tensor2D.h ../include/Tensor1D.h \
 ../include/TensorFunction.h

testconvert.o: testconvert.cxx ../include/convert.h \
 ../include/NumericInfo.h ../include/pet_common.h \
 ../include/Tensor1D.h ../include/NumericVectorWithOffset.h \
 ../include/VectorWithOffset.h ../include/Tensor2D.h \
 ../include/Tensorbase.h ../include/Tensor3D.h ../include/Tensor4D.h


clean:
	rm tensor display sinodata VectorWithOffset convert \
	 displayPSOV ParaEPXtest pplustiny pplusdummy ecat6bysegm \
	ecat6bysegmspan libcti_utils.a libreconstruction.a libfft.a\
	libs_cti/*o
