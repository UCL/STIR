# $Id$ : $Date$
CXX = g++
CC = gcc
PXCXX = px g++
#KT 28/11 some new defines
WORKSPACE = ${HOME}/PPhead
INCLUDE = ${WORKSPACE}/include
DISPLAYLIB = ${WORKSPACE}/display/libdisplay.a
BUILDBLOCKLIB = ${WORKSPACE}/buildblock/libbuildblock.a
PARAPLUSLIB = /usr/local/lib/libparaplus.a

#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
CFLAGS = -g -D_DEBUG -I${INCLUDE} -I/usr/openwin/include -DTEST -L/usr/local/lib
#KT 28/11 added buildblock library
#KT 28/11 added the last -L option for Solaris
LDFLAGS = -g -L${WORKSPACE}/display  -L${WORKSPACE}/buildblock \
          -L/usr/openwin/lib -lparaplus \
          -lbuildblock -ldisplay -lX11 -lcurses

%.o:%.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
#KT GNU make would allow $+ in the rule below to include all dependents, 
#but Solaris make doesn't...
%:%.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ $< ${LDFLAGS}


all: tensor display sinodata VectorWithOffset convert paraplus

paraplus:	ParaEPXtest pplustiny pplusdummy

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


VectorWithOffset: VectorWithOffset.cxx ${INCLUDE}/VectorWithOffset.h

convert: convert.cxx ${BUILDBLOCKLIB}
	$(CXX) $(CFLAGS) -o $@ $< 

pplusdummy: pplusdummy.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $< -lparaplus

pplustiny: pplustiny.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $< -lparaplus

ParaEPXtest: ParaEPXtest.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $< -lparaplus

