# $Id$ : $Date$

CXX = g++
CC = gcc
#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -D__unix__ \
-I/usr/openwin/include
LDFLAGS = -g -L${HOME}/PPhead/display -lX11 -lcurses

%.o:%.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%.o:%.c
	$(CC) $(DEBUG) $(CFLAGS) -o $@ -c $< 


libdisplay.a: display.o gen.o screengen.o screen.o
	ar r $@ $?
	ranlib $@

