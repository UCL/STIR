CXX = g++
CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEST
#CXX = xlC
#CFLAGS = -g -D_DEBUG -I${HOME}/PPhead/include -DTEST -DTEMPLATE_ARG

%.o:%.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%:%.o
	$(CXX) $(DEBUG) $(CFLAGS) -o $@  $< 


sinodata: sinodata.o

sinodata.o : sinodata.cxx

test: test.cc
