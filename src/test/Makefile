#
# $Id$: $Date$
# 

include ../Makefile_common
include ../Makefile_bblibs

# TODO enable test_coordinates

all: $(DEST)tensor $(DEST)display2 \
	$(DEST)VectorWithOffset $(DEST)testconvert \
	$(DEST)test_linear_regression \
	$(DEST)test_interpolate \
	$(DEST)test_filename_functions 
#	$(DEST)test_coordinates

get_test_files: input/test_linear_regression.in

run_tests: all get_test_files
	$(DEST)VectorWithOffset
	$(DEST)tensor
	$(DEST)testconvert
	$(DEST)test_linear_regression  input/test_linear_regression.in
	$(DEST)test_interpolate
	$(DEST)test_filename_functions 
#	$(DEST)test_coordinates


.PHONY: run_tests get_test_files

# disable default rule
# as executables here do not depend on recon_buildblock
${DEST}%: ${DEST}%.o $(BB_LIBS)

${DEST}%: ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          ${LIB_DISPLAY} ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS)
