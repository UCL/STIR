# $Id$ : $Date$
CXX = g++
CFLAGS = -g -D_DEBUG -I../include 

%.o:%.cxx
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 
%.o:%.cc
	$(CXX) $(DEBUG) $(CFLAGS) -o $@ -c $< 

#KT just deleted the -DOTHER_TESTS option here
tensor: tensor.cxx
	$(CXX) $(CFLAGS) -o $@ $<

# DO NOT DELETE THIS LINE -- g++dep uses it.
# DO NOT PUT ANYTHING AFTER THIS LINE, IT WILL GO AWAY.
# IF YOU PUT ANYTHING HERE IT WILL GO AWAY
