# AIX Makefile of the PPhead package
# 
#@(#)Makefile	1.3 : 00/01/13
#
#CL Remove old version promis_span and main_promis


WORKSPACE = ${HOME}/PPhead
INCLUDE_DIR = ${WORKSPACE}/include

#These libraries will be used as dependencies
LIB_RECON_BUILDBLOCK= \
   ${WORKSPACE}/recon_buildblock/${DEST}librecon_buildblock.a
LIB_BUILDBLOCK= \
   ${WORKSPACE}/buildblock/${DEST}libbuildblock.a
LIB_DISPLAY= \
   ${WORKSPACE}/display/${DEST}libdisplay.a
#CL 02/98/98 Add this parameter 
ALL_LIBS = $(LIB_RECON_BUILDBLOCK) $(LIB_BUILDBLOCK) $(LIB_DISPLAY)

SUBDIRECTORY = $(WORKSPACE)/analytic/PROMIS\
	$(WORKSPACE)/iterative/OSEM \
	$(WORKSPACE)/utilities \
	$(WORKSPACE)/recon_test \
	$(WORKSPACE)/test 

LIBDIRECTORY = $(WORKSPACE)/display\
	$(WORKSPACE)/recon_buildblock \
	$(WORKSPACE)/buildblock \

#all the others directories will be activated later
#$(WORKSPACE)/analytic/FORE $(WORKSPACE)/iterative/COSEM $(WORKSPACE)/sparse_test $(WORKSPACE)/parallel


DEBUG= -D_DEBUG 
DEST=debug/
CXX= g++ 

ALL = "DEBUG=$DEBUG DEST=$DEST all"

#KT 22/05/98 use INCLUDE_DIR and DEBUG and more warnings
CFLAGS= -g $(DEBUG) -Decho -I$(INCLUDE_DIR) -I/usr/lib -Wconversion -Wstrict-prototypes 

MAKE_ALL = $(MAKE) all

RM = /bin/rm -f 

LIBS1 = -lm
LIBS = $(LIBS1)  -lcurses -lX11
#For Solaris LIBS_SOL = $(LIBS1) -L/usr/openwin/lib -lcurses -lX11 -lnsl -lsocket
# For PARA LIBS_AIX = -lcurses -lX11



.SUFFIXES: .cc .o .cxx .a

.SCCS_GET:

.KEEP_STATE:

.PRECIOUS: $(LIB_DISPLAY) $(LIB_BUILDBLOCK) $(LIB_RECON_BUILDBLOCK)

createdir: 
	-mkdir -p $(DEST)

all: $(LIB_DISPLAY) $(LIB_BUILDBLOCK) $(LIB_RECON_BUILDBLOCK)
	for subdir in $(SUBDIRECTORY); do   \
	  	(echo "------------------");\
		(cd $$subdir; $(MAKE_ALL)); \
	done



# KT 13/01/2000 inserted ; \ after first done
clean: 
#	echo -n "Are you sure to clean up all the directories ? (y/n)"
#	set resp = $<
#	if ($resp == 'y') then
		for subdir in $(SUBDIRECTORY); do \
	  		(echo \rm -rf $$subdir/$(DEST)); \
			(cd $$subdir; \rm -r $(DEST));\
		done; \
		for libdir in $(LIBDIRECTORY); do \
	  		(echo \rm -rf $$libdir/$(DEST)); \
			(cd $$libdir; \rm -r $(DEST));\
		done
#	else
#		(echo "No clean up ");
#	fi



#CL 09/06/98 Add librairies, the macro FORCE force a nested make command to run,
#regardless of the file's presence, by making it depend on a target with a null rule

$(LIB_DISPLAY) : FORCE 
	cd $(WORKSPACE)/display; $(MAKE)

$(LIB_BUILDBLOCK) : FORCE
	cd $(WORKSPACE)/buildblock; $(MAKE)

$(LIB_RECON_BUILDBLOCK) : FORCE
	cd $(WORKSPACE)/recon_buildblock; $(MAKE) 

FORCE:
