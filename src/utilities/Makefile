#
# $Id$ :$Date$
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include

#These libraries will be used as dependencies
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

${DEST}% : ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
#.o:
	$(CXX) $(CFLAGS) -o $@ $< \
	    ${LIB_BUILDINGBLOCK} \
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



# KT 09/10/98 changed to *.o
.PRECIOUS: $(DEST)*.o

# KT 09/10/98 use EXECS to prevent repetition, add displayIF
EXECS =  $(DEST)vox $(DEST)displayPSOV $(DEST)swap_byteorder $(DEST)displayIF


all: createdir libs $(EXECS)

createdir:
	-mkdir -p $(DEST)

libs: $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)

$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE);

# KT 09/10/98 changed spelling mistake BUILD -> BUILDING
$(LIB_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); 

FORCE:


clean: 
	$(RM) $(DEST)*.o $(EXECS)
