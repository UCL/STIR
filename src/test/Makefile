#
# $Id$: $Date$
# 

include ../Makefile_common

# TODO ? differentiate between those that need graphics and those that don't
SOURCES = test_Array.cxx \
	test_VectorWithOffset.cxx \
        test_convert_array.cxx \
	test_IndexRange.cxx \
	test_coordinates.cxx \
	test_linear_regression.cxx \
	test_interpolate.cxx \
	test_filename_functions.cxx \
	test_coordinates.cxx \
	test_VoxelsOnCartesianGrid.cxx \
	test_proj_data_info.cxx \
	test_display.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

include $(SOURCES:%.cxx=$(DEST)%.d)

install:
	echo "Not installing any executables from test"

clean:
	$(RM) $(DEST)*

include ../Makefile_bblibs

get_test_files: input/test_linear_regression.in

run_tests: all get_test_files
	$(DEST)test_VectorWithOffset
	$(DEST)test_Array
	$(DEST)test_convert_array
	$(DEST)test_IndexRange
	$(DEST)test_filename_functions
	$(DEST)test_linear_regression  input/test_linear_regression.in
	$(DEST)test_coordinates
	$(DEST)test_filename_functions 
	$(DEST)test_VoxelsOnCartesianGrid
	$(DEST)test_proj_data_info

run_interactive_tests: all
	$(DEST)test_interpolate
	$(DEST)test_display

.PHONY: run_tests run_interactive_tests get_test_files

.PRECIOUS:  $(OBJS) $(EXES)

# Disable default rule
# as executables here do not depend on recon_buildblock
${DEST}%: ${DEST}%.o $(BB_LIBS)

${DEST}%: ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          ${LIB_DISPLAY} ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)




