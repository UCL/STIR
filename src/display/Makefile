# AIX and SOLARIS Makefile for PPhead/display directory
#
# $Id$
#

# DB 1/2/99 Add WORKSPACE and INCLUDE_DIR directory
WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include

CXX = g++
CC = gcc
DEBUG= -D_DEBUG
DEST=debug/
#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
#CL 030698 Move debug flags to DEBUG
#__unix__ has to be defined for gen.h 
# DB 1/2/99 change CFLAGS to include -I$(INCLUDE_DIR)

CFLAGS =  -g $(DEBUG) -I$(INCLUDE_DIR) -D__unix__ \
-I/usr/openwin/include

#KT 010798 removed as we do not link in this makefile
#LDFLAGS = -g -L${HOME}/PPhead/display -lX11 -lcurses

.KEEP_STATE:

#CL060798 Add createdir for automtaically create directories

all: createdir ${DEST}libdisplay.a 


createdir: 
	-mkdir -p $(DEST)




${DEST}%.o: %.cxx
	$(CXX)  $(CFLAGS) -o $@ -c $< 

${DEST}%.o: %.cc
	$(CXX)  $(CFLAGS) -o $@ -c $< 

${DEST}%.o: %.c 
	$(CC)  $(CFLAGS) -o $@ -c $< 

#CL 09/10/98 ADd the header file dependencies for all object files
#I agree that this Makefile become big but it is the only way to update correctly 
#all the object files

$(DEST)display.o: display.cxx ../include/VectorWithOffset.h \
 ../include/pet_common.h ../include/Tensor2D.h ../include/Tensorbase.h \
 ../include/NumericVectorWithOffset.h ../include/NumericInfo.h \
 ../include/Tensor1D.h ../include/Tensor3D.h ../include/display.h \
 ../include/sinodata.h ../include/PETScannerInfo.h \
 ../include/PETScanInfo.h gen.h screen.h

$(DEST)gen.o: gen.c gen.h

$(DEST)screen.o: screen.c gen.h screen.h

$(DEST)screengen.o: screengen.c gen.h screen.h

.PRECIOUS: ${DEST}*.a ${DEST}*.o



${DEST}libdisplay.a: \
    ${DEST}display.o ${DEST}gen.o ${DEST}screengen.o ${DEST}screen.o
	ar r $@ $?
	ranlib $@

