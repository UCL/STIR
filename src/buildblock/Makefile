#
# $Id$: $Date$
#


CXX = g++
RM = /bin/rm -f 
# KT 18/12/97 removed Para++ related CFLAGS
#CL030698 Move debug flags to DEBUG
DEBUG=-D_DEBUG
DEST=debug/
CFLAGS = -g $(DEBUG) -I${HOME}/PPhead/include
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include
#CXX = xlC
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEMPLATE_ARG

#KT 27/02/98 use default rules now, .cxx does not work without next line
.SUFFIXES: .cc .o .cxx .a

.KEEP_STATE:

#KT 09/10/98 added interfile to buildblock
all: createdir ${DEST}libbuildblock.a

createdir: 
	-mkdir -p $(DEST)


#KT
${DEST}%.o : %.cxx 
#.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 


${DEST}libbuildblock.a: ${DEST}sinodata.o ${DEST}convert.o \
   ${DEST}PETScannerInfo.o ${DEST}Tensor1D.o ${DEST}viewdata.o \
   $(DEST)utilities.o $(DEST)NumericInfo.o \
   ${DEST}KeyParser.o ${DEST}line.o ${DEST}interfile.o ${DEST}bbfilters.o
	ar r $@ $?
	ranlib $@


#KT 30/07/98 added DEST
clean: 
	$(RM) $(DEST)*.o
	$(RM) $(DEST)*.a
