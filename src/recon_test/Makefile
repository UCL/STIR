# AIX Makefile for PPhead/recon_test directory

# $Id$: $Date$
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include

#These libraries will be used as dependencies
LIB_RECON_BUILDINGBLOCK= \
   ${WORKSPACE}/recon_buildblock/${DEST}librecon_buildblock.a
LIB_BUILDINGBLOCK= \
   ${WORKSPACE}/buildblock/${DEST}libbuildblock.a
LIB_DISPLAY= \
   ${WORKSPACE}/display/${DEST}libdisplay.a



DEBUG= -D_DEBUG
DEST=debug/
CXX= g++ 


#KT 22/05/98 use INCLUDE_DIR and DEBUG and more warnings
CFLAGS= -g $(DEBUG) -Decho -I$(INCLUDE_DIR) -Wconversion -Wstrict-prototypes 
#LDFLAGS= -Iinclude 

.SUFFIXES: .cc .o .cxx .a

${DEST}%.o : %.cxx
#.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 


${DEST}% : ${DEST}%.o ${LIB_RECON_BUILDINGBLOCK} ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
#.o:
	$(CXX) $(CFLAGS) -o $@ $< \
	    ${LIB_RECON_BUILDINGBLOCK} ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}


RM = /bin/rm -f 

LIBS1 = -lm

#KT 22/05/98 use GNU make specific syntax to differentiate between systems (sorry)
ifdef CYGROOT
#Cygwin32 doesn't have curses
LIBS = $(LIBS1) -L/usr/X11R6.3/lib -lX11 
else
LIBS = $(LIBS1)  -lX11 -lcurses
endif
#LIBS_SOL = -L/usr/openwin/lib -L../../../LIBECAT/SOL2  -lm -lecat -lnsl -lsocket

.KEEP_STATE:

#specify files which should not be deleted at the end of a make
#KT 05/10/98 replaced = sign with : in next line, works now
.PRECIOUS: ${DEST}bcktest.o ${DEST}fwdtest.o ${DEST}zoomtest.o \
    $(LIB_DISPLAY) $(LIB_BUILDBLOCK) $(LIB_RECON_BUILDINGBLOCK)

#KT 05/10/98 added createdir and FORCE to list of targets
# So, the corresponding commands will always be executed 
# (when doing 'make' or 'make all')
all: createdir FORCE ${DEST}bcktest ${DEST}fwdtest ${DEST}zoomtest

createdir: 
	-mkdir -p $(DEST)


#zoomtest:  zoomtest.o $(LIB_RECON_BUILDINGBLOCK)
#	$(CXX) $(CFLAGS) -o $@ $< ${LIBS}




$(LIB_RECON_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/recon_buildblock; $(MAKE);

$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE);

$(LIB_BUILDBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); 

FORCE:

