# $Id$ : $Date$
CXX = g++
CC = gcc
PXCXX = px g++

WORKSPACE = ${HOME}/PPhead
#KT 02/03/98 changed INCLUDE to INCLUDE_DIR to avoid conflict with Visual C++ environment variable
INCLUDE_DIR = ${WORKSPACE}/include
DISPLAYLIB = ${WORKSPACE}/display/libdisplay.a
#KT 22/01/98 added this variable to differentiate between programs who use display
# the 2nd -L option is for Solaris
DISPLAYSWITCHES =  -L${WORKSPACE}/display  \
          -L/usr/openwin/lib -ldisplay -lX11 -lcurses
BUILDBLOCKLIB = ${WORKSPACE}/buildblock/libbuildblock.a
PARAPLUSLIB =  /usr/local/lib/libparaplus.a
# KT 16/12 added interfile
INTERFILELIB = ${WORKSPACE}/buildblock/libinterfile.a

#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
# KT 02/03/98 removed -DTEST option
DEBUG = -g -D_DEBUG
CFLAGS = ${DEBUG} -I ${INCLUDE_DIR} -I /usr/openwin/include
LDFLAGS = -g -L${WORKSPACE}/buildblock  -lbuildblock

# KT 10/03/98 use default rules
.SUFFIXES: .o .c .cxx .cc
.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 
.cc.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 
#KT GNU make would allow $+ in the rule below to include all dependents, 
#but Solaris make doesn't...
.cxx:
	$(CXX) $(CFLAGS) -o $@ $< ${LDFLAGS}


# KT 10/03/98 use 2 targets for parallel programs and non-parallel ones

all: paraplus nopara

# KT 05/05/98 changed convert to testconvert
nopara: tensor display sinodata VectorWithOffset testconvert \
	 displayPSOV

paraplus:	ParaEPXtest pplustiny pplusdummy

#KT added .h dependencies
tensor: tensor.cxx ${INCLUDE_DIR}/Tensor1D.h ${INCLUDE_DIR}/Tensor2D.h \
	${INCLUDE_DIR}/Tensor3D.h ${INCLUDE_DIR}/Tensor4D.h \
	${INCLUDE_DIR}/Tensorbase.h ${INCLUDE_DIR}/VectorWithOffset.h

#KT The -I${WORKSPACE}/display flag is only necessary because I used gen.h
#(part of my old library). 
display: display.cxx ${DISPLAYLIB} 
	$(CXX) $(CFLAGS) -I${WORKSPACE}/display -o $@ $< $(LDFLAGS) \
	${DISPLAYSWITCHES}

sinodata: sinodata.cxx ${DISPLAYLIB} ${BUILDBLOCKLIB} 
	$(CXX) $(CFLAGS) -o $@ $<  $(LDFLAGS) \
	${DISPLAYSWITCHES}

# KT 2/03/98 added
#KT The -I${WORKSPACE}/display flag is only necessary because I used gen.h
#(part of my old library). 
displayPSOV: displayPSOV.cxx ${DISPLAYLIB} ${BUILDBLOCKLIB} 
	$(CXX) $(CFLAGS) -o $@  -I${WORKSPACE}/display $<  $(LDFLAGS)  \
	${DISPLAYSWITCHES}

VectorWithOffset: VectorWithOffset.cxx ${INCLUDE_DIR}/VectorWithOffset.h

testconvert: testconvert.cxx ${BUILDBLOCKLIB}
 
testinterfilereader: testinterfilereader.cxx ${INTERFILEFLIB}
	$(CXX) $(CFLAGS) -o $@ $<  $(LDFLAGS) ${INTERFILELIB}

pplusdummy: pplusdummy.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)

pplustiny: pplustiny.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)

ParaEPXtest: ParaEPXtest.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)


clean:
	rm tensor display sinodata VectorWithOffset convert \
	 displayPSOV ParaEPXtest pplustiny pplusdummy
