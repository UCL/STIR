#
# $Id$: $Date$
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include


DEBUG= -D_DEBUG
CXX= g++ 
SYS = 

#KT 22/05/98 use INCLUDE_DIR and DEBUG and more warnings
CFLAGS= -g $(DEBUG) -I$(INCLUDE_DIR) -Wconversion -Wstrict-prototypes 
#LDFLAGS= -Iinclude 

#KT 22/05/98 use 'old style' default rule
.SUFFIXES: .cc .o .cxx .a

.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 

.cxx:
	$(CXX) $(CFLAGS) -o $@ $< ${LIBS}


RM = /bin/rm -f 
#KT 22/05/98 use WORKSPACE and add recon_buildblock
LIBS1 = -L$(WORKSPACE)/display/$(SYS) -L$(WORKSPACE)/buildblock/$(SYS) \
 -L$(WORKSPACE)/recon_buildblock/$(SYS) \
  -lm  -ldisplay -lbuildblock -lrecon_buildblock 

#KT 22/05/98 use GNU make specific syntax to differentiate between systems (sorry)
ifdef CYGROOT
#Cygwin32 doesn't have curses
LIBS = $(LIBS1) -L/usr/X11R6.3/lib -lX11 
else
LIBS = $(LIBS1)  -lX11 -lcurses
endif
#LIBS_SOL = -L/usr/openwin/lib -L../../../LIBECAT/SOL2  -lm -lecat -lnsl -lsocket




.KEEP_STATE:

all: bcktest fwdtest zoomtest

