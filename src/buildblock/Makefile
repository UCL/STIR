#
# @(#)Makefile	1.11: 98/07/02
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

#KT 16/12 wanted to add interfile...
#CL 060798 Add createdir for create directories of debugged or optimized files versions
all: createdir ${DEST}libbuildblock.a #${DEST}libinterfile.a

createdir: 
	-mkdir -p $(DEST)


#KT
${DEST}%.o : %.cxx 
#.cxx.o:
	$(CXX) $(CFLAGS) -o $@ -c $< 



# KT 30/07/98 added utilities
${DEST}libbuildblock.a: ${DEST}sinodata.o ${DEST}convert.o \
   ${DEST}PETScannerInfo.o ${DEST}Tensor1D.o ${DEST}viewdata.o \
   $(DEST)utilities.o
	ar r $@ $?
	ranlib $@

${DEST}libinterfile.a: ${DEST}KeyParser.o ${DEST}line.o ${DEST}pet_string.o

#KT 30/07/98 added DEST
clean: 
	$(RM) $(DEST)*.o
	$(RM) $(DEST)*.a
