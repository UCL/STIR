#
# @(#)Makefile	1.1: 98/05/22
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include


DEBUG= -D_DEBUG
CXX= g++ 
SYS = 

#KT 22/05/98 use INCLUDE_DIR and DEBUG and more warnings
CFLAGS= -g $(DEBUG) -Decho -I$(INCLUDE_DIR) -Wconversion -Wstrict-prototypes 
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

#ADd this library in order to update automatically when zoomtest is compiled
LIB_RECON_BUILDINGBLOCK= ${WORKSPACE}/recon_buildblock/librecon_buildblock.a


.KEEP_STATE:

.PRECIOUS = $(LIB_DISPLAY) $(LIB_BUILDBLOCK) $(LIB_RECON_BUILDINGBLOCK)


all: bcktest fwdtest zoomtest


zoomtest:  zoomtest.o $(LIB_RECON_BUILDINGBLOCK)
	$(CXX) $(CFLAGS) -o $@ $< ${LIBS}



$(LIB_RECON_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/recon_buildblock; $(MAKE); \cp *[oa] $(SYS)

$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE); \cp *[oa] $(SYS)

$(LIB_BUILDBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); \cp *[oa] $(SYS)

FORCE:

