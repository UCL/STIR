#
# $Id$ : $Date$
#

WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include
RANLIB= ranlib

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

CFLAGS= -g $(DEBUG) -Decho -I$(INCLUDE_DIR) -Wconversion -Wstrict-prototypes 

.SCCS_GET:

.SUFFIXES: .cc .o .cxx .a


RM = /bin/rm -f 

LIBS1 = -lm

LIBS = $(LIBS1) -L/usr/X11/lib -lX11 -lcurses

SOURCES = vox.cxx binmath.cxx displayPSOV.cxx convif_ecat6.cxx convecat6_if.cxx \
	correctPSOV.cxx postfilter.cxx swap_byteorder.cxx
OBJS=$(SOURCES:%.cxx=$(DEST)%.o)
EXES=$(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS)

all: createdir libs $(EXES) 
 
createdir:
	-mkdir -p $(DEST)

$(DEST)libcti.a: $(DEST)cti_utils.o
	$(AR) r $@ $?
	$(RANLIB) $@

$(DEST)convif_ecat6: convif_ecat6.cxx $(DEST)libcti.a ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< -L$(DEST) -lcti -lm \
		${LIB_BUILDINGBLOCK} 


$(DEST)convecat6_if: convecat6_if.cxx $(DEST)libcti.a ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< -L$(DEST) -lcti -lm \
		${LIB_BUILDINGBLOCK} 

$(DEST)correctPSOV: correctPSOV.cxx ${LIB_RECON_BUILDBLOCK} $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)
	$(CXX) $(CFLAGS)  -o $@ $< \
		${LIB_RECON_BUILDBLOCK} \
		${LIB_BUILDINGBLOCK} \
		${LIB_DISPLAY} ${LIBS}

$(DEST)swap_byteorder: swap_byteorder.cxx ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< \
		${LIB_BUILDINGBLOCK}

libs: $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)


$(LIB_DISPLAY): FORCE
	cd $(WORKSPACE)/display; $(MAKE);

$(LIB_BUILDINGBLOCK): FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE); 

$(LIB_RECON_BUILDBLOCK) : FORCE
	cd $(WORKSPACE)/recon_buildblock; $(MAKE)

$(DEST)%.o : %.cxx
	$(CXX) $(CFLAGS) -o $@ -c $< 

$(DEST)% : $(DEST)%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS) -o $@ $< \
	    ${LIB_BUILDINGBLOCK} \
	    ${LIB_DISPLAY} ${LIBS}

FORCE:

clean: 
	$(RM) $(DEST)*.o $(EXES)
