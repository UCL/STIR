# @(#)Makefile	$Id$
CXX = g++
CC = gcc
DEBUG= -D_DEBUG
#KT added the last -I option for Solaris, doesn't harm anyone if it's
# not necessary
#CL 030698 Move debug flags to DEBUG

CFLAGS =  -g $(DEBUG) -I${HOME}/PPhead/include -D__unix__ \
-I/usr/openwin/include
LDFLAGS = -g -L${HOME}/PPhead/display -lX11 -lcurses

%.o:%.cxx
	$(CXX)  $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX)  $(CFLAGS) -o $@ -c $< 
%.o:%.c
	$(CC)  $(CFLAGS) -o $@ -c $< 


libdisplay.a: display.o gen.o screengen.o screen.o
	ar r $@ $?
	ranlib $@

