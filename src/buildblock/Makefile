CXX = g++
RM = /bin/rm -f 
# KT 18/12/97 removed Para++ related CFLAGS
CFLAGS = -O3 -I${HOME}/PPhead/include
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include
#CXX = xlC
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEMPLATE_ARG

#KT 27/02/98 use default rules now, .cxx does not work without next line
.SUFFIXES: .cc .o .cxx .a

#KT 16/12 wanted to add interfile...
all:	libbuildblock.a #libinterfile.a

#KT 27/02/98 use default rules now
.cxx.o:
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 



libbuildblock.a: sinodata.o convert.o PETScannerInfo.o Tensor1D.o viewdata.o
	ar r $@ $?
	ranlib $@
libinterfile.a: KeyParser.o line.o pet_string.o

#KT 27/02/98 use default rules now, so all this can go
#convert.o: convert.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

#Tensor1D.o: Tensor1D.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

#sinodata.o: sinodata.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

#PETScannerInfo.o:  PETScannerInfo.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

#KT 16/12 added next 3

KeyParser.o:  KeyParser.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

line.o:  line.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

pet_string.o:  pet_string.cxx
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

clean: 
	$(RM) *.o
	$(RM) *.a
