#
# $Id$: $Date$
# 
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common

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
	test_display.cxx \
	test_stir_math.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


install:
	@echo "Not installing any executables from test"

clean:
	$(RM) $(DEST)*

include $(WORKSPACE)/Makefile_bblibs

get_test_files: input/test_linear_regression.in

run_tests: all get_test_files make_stir_math
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
	$(DEST)test_stir_math ../utilities/$(DEST)stir_math

# running test_stir_math depends on the stir_math executable itself
make_stir_math:
	$(MAKE) -C ../utilities $(DEST)stir_math
    
run_interactive_tests: all
	$(DEST)test_interpolate
	$(DEST)test_display

.PHONY: run_tests run_interactive_tests get_test_files make_stir_math

.PRECIOUS:  $(OBJS) $(EXES)


# Disable default rule
# as executables here do not depend on recon_buildblock or the registries
${DEST}%: ${DEST}%.o $(BB_LIBS) $(BB_REGISTRIES)


${DEST}%: ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_DISPLAY}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          ${LIB_DISPLAY} ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)




