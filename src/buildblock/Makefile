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

#KT 16/12 wanted to add interfile...
all:	${DEST}libbuildblock.a #${DEST}libinterfile.a

#KT
${DEST}%.o : %.cxx 
#.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 



${DEST}libbuildblock.a: ${DEST}sinodata.o ${DEST}convert.o \
   ${DEST}PETScannerInfo.o ${DEST}Tensor1D.o ${DEST}viewdata.o
	ar r $@ $?
	ranlib $@

${DEST}libinterfile.a: ${DEST}KeyParser.o ${DEST}line.o ${DEST}pet_string.o

clean: 
	$(RM) *.o
	$(RM) *.a
