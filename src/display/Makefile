# @(#)Makefile	$Id$
CXX = g++
CC = gcc
DEBUG= -D_DEBUG
DEST=debug/
#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
#CL 030698 Move debug flags to DEBUG
#__unix__ has to be defined for gen.h
CFLAGS =  -g $(DEBUG) -I${HOME}/PPhead/include -D__unix__ \
-I/usr/openwin/include
#KT 010798 removed as we do not link in this makefile
#LDFLAGS = -g -L${HOME}/PPhead/display -lX11 -lcurses

${DEST}%.o : %.cxx
#.cxx.o:
	$(CXX)  $(CFLAGS) -o $@ -c $< 
${DEST}%.o:%.cc
	$(CXX)  $(CFLAGS) -o $@ -c $< 
${DEST}%.o:%.c
	$(CC)  $(CFLAGS) -o $@ -c $< 

.PRECIOUS: ${DEST}*.a ${DEST}*.o

${DEST}libdisplay.a: \
    ${DEST}display.o ${DEST}gen.o ${DEST}screengen.o ${DEST}screen.o
	ar r $@ $?
	ranlib $@

