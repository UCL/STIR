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
	test_stir_math.cxx \
	test_OutputFileFormat.cxx \
 	test_ByteOrder.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif

install:
	@echo "Not installing any executables from test"

clean:
	$(RM) $(DEST)*

include $(WORKSPACE)/Makefile_bblibs

get_test_files: input/test_linear_regression.in

run_tests: all get_test_files make_stir_math run_tests_ecat7
	$(DEST)test_VectorWithOffset
	$(DEST)test_Array
	$(DEST)test_convert_array
	$(DEST)test_IndexRange
	$(DEST)test_filename_functions
	$(DEST)test_linear_regression  input/test_linear_regression.in
	$(DEST)test_coordinates
	$(DEST)test_VoxelsOnCartesianGrid
	$(DEST)test_proj_data_info
	$(DEST)test_stir_math ../utilities/$(DEST)stir_math 
	$(DEST)test_OutputFileFormat input/test_InterfileOutputFileFormat.in 
	$(DEST)test_OutputFileFormat input/test_InterfileOutputFileFormat_short.in 
	$(DEST)test_OutputFileFormat input/test_ECAT6OutputFileFormat.in
	$(DEST)test_ByteOrder

ifeq ($(HAVE_LLN_MATRIX),1)
run_tests_ecat7:
	$(DEST)test_OutputFileFormat input/test_ECAT7OutputFileFormat.in

else
run_tests_ecat7:
	@echo No ECAT7 support compiled, so no tests for this file format
endif

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


${DEST}%: ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_IO} ${LIB_DISPLAY} ${IO_REGISTRIES}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          $(IO_REGISTRIES) \
          ${LIB_DISPLAY} ${LIB_IO} ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)




