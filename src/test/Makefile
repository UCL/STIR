#
# $Id$: $Date$
# 

include ../Makefile_common

# TODO ? differentiate between those that need graphics and those that don't
SOURCES = tensor.cxx display2.cxx \
	VectorWithOffset.cxx testconvert.cxx \
	test_linear_regression.cxx \
	test_interpolate.cxx \
	test_filename_functions.cxx \
	test_coordinates.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

include $(SOURCES:%.cxx=$(DEST)%.d)

clean:
	$(RM) $(DEST)*

include ../Makefile_bblibs

get_test_files: input/test_linear_regression.in

run_tests: all get_test_files
	$(DEST)VectorWithOffset
	$(DEST)tensor
	$(DEST)testconvert
	$(DEST)test_linear_regression  input/test_linear_regression.in
	$(DEST)test_interpolate
	$(DEST)test_coordinates
	$(DEST)test_filename_functions 

.PHONY: run_tests get_test_files

# disable default rule
# as executables here do not depend on recon_buildblock
${DEST}%: ${DEST}%.o $(BB_LIBS)

${DEST}%: ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          ${LIB_DISPLAY} ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)




