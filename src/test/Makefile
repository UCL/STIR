# $Id$ : $Date$
CXX = g++
CC = gcc
#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
WORKSPACE = ${HOME}/PPhead
INCLUDE = ${WORKSPACE}/include

CFLAGS = -g -D_DEBUG -I${INCLUDE} -I/usr/openwin/include
LDFLAGS = -g -L${WORKSPACE}/display  -ldisplay -lX11 -lcurses

%.o:%.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

all: tensor display sinodata

#KT added .h dependencies
tensor: tensor.cxx ${INCLUDE}/Tensor1D.h ${INCLUDE}/Tensor2D.h \
	${INCLUDE}/Tensor3D.h ${INCLUDE}/Tensor4D.h \
	${INCLUDE}/Tensorbase.h ${INCLUDE}/VectorWithOffset.h
	$(CXX) $(CFLAGS) -o $@ $<

#KT The -I${WORKSPACE}/display flag is only necessary because I used gen.h
#(part of my old library). 
display: display.cxx ${WORKSPACE}/display/libdisplay.a
	$(CXX) $(CFLAGS) -I${WORKSPACE}/display -o $@ $< $(LDFLAGS)

sinodata: sinodata.cxx ${WORKSPACE}/buildblock/sinodata.o
	$(CXX) $(CFLAGS) -o $@ $< ${WORKSPACE}/buildblock/sinodata.o $(LDFLAGS)


