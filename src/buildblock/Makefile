CXX = g++
CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEST -I/epx/epx/include -D_CR_LIB_
#CXX = xlC
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEST -DTEMPLATE_ARG

all:	ParaProcess.o ParaProcessPattern.o ParaStream.o ParaStreamIn.o ParaStreamOut.o ParaStreamInOps.o ParaStreamOutOps.o ParaEPX.o libbuildblock.a

#%.o:	%.cxx
#.o.cxx:
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
#%.o:	%.cc
#.o.cc:
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
#%:	%.o
#..o:
#	$(CXX) $(DEBUG) $(CFLAGS) -o $@  $< 


libbuildblock.a: sinodata.o convert.o # Tensor1D.o
	ar r $@ $?
	ranlib $@

convert.o: convert.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

Tensor1D.o: Tensor1D.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

sinodata.o: sinodata.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaProcess.o:	ParaProcess.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaProcessPattern.o: ParaProcessPattern.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaStream.o:	ParaStream.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaStreamIn.o:	ParaStreamIn.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaStreamOut.o:	ParaStreamOut.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaStreamInOps.o:	ParaStreamInOps.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaStreamOutOps.o:	ParaStreamOutOps.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

ParaEPX.o:	ParaEPX.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

