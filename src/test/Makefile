# $Id$: $Date$
# KT 28/07/98 use DEST syntax as in other places
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
PXCXX = px g++


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

# From here specific to this Makefile

PARAPLUSLIB =  /usr/local/lib/libparaplus.a
INTERFILELIB = ${WORKSPACE}/buildblock/$(DEST)/libinterfile.a

# KT 10/03/98 use 2 targets for parallel programs and non-parallel ones

all: paraplus nopara

# KT 05/05/98 changed convert to testconvert
nopara: $(DEST)tensor $(DEST)display $(DEST)sinodata \
	$(DEST)VectorWithOffset $(DEST)testconvert \
	$(DEST)displayPSOV

paraplus:	$(DEST)ParaEPXtest $(DEST)pplustiny $(DEST)pplusdummy

# display still uses ask() etc from gen.h (TODO)
$(DEST)display.o: display.cxx 
	$(CXX) $(CFLAGS) -c -o $@ $<  -I${WORKSPACE}/display 

$(DEST)testinterfilereader: testinterfilereader.cxx ${INTERFILEFLIB}
	$(CXX) $(CFLAGS) -o $@ $<  $(LDFLAGS) ${INTERFILELIB}

$(DEST)pplusdummy: pplusdummy.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)

$(DEST)pplustiny: pplustiny.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)

$(DEST)ParaEPXtest: ParaEPXtest.cxx $(PARAPLUSLIB)
	$(PXCXX) -o $@ $(CFLAGS) $<  $(PARAPLUSLIB)


clean:
	rm $(DEST)tensor $(DEST)display $(DEST)sinodata \
           $(DEST)VectorWithOffset $(DEST)testconvert \
	   $(DEST)displayPSOV \
	   $(DEST)ParaEPXtest $(DEST)pplustiny $(DEST)pplusdummy
