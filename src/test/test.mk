#
# $Id$
# 
dir := test

$(dir)_TEST_SOURCES := test_Array.cxx \
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


run_tests_$(dir): all_test_exes  $(DEST)utilities/stir_math
	$(DEST)test/test_VectorWithOffset
	$(DEST)test/test_Array
	$(DEST)test/test_convert_array
	$(DEST)test/test_IndexRange
	$(DEST)test/test_filename_functions
	$(DEST)test/test_linear_regression  test/input/test_linear_regression.in
	$(DEST)test/test_coordinates
	$(DEST)test/test_VoxelsOnCartesianGrid
	$(DEST)test/test_proj_data_info
	$(DEST)test/test_stir_math $(DEST)/utilities/stir_math 
	$(DEST)test/test_OutputFileFormat test/input/test_InterfileOutputFileFormat.in 
	$(DEST)test/test_OutputFileFormat test/input/test_InterfileOutputFileFormat_short.in 
	$(DEST)test/test_OutputFileFormat test/input/test_ECAT6OutputFileFormat.in
	$(DEST)test/test_ByteOrder
ifeq ($(HAVE_LLN_MATRIX),1)
	$(DEST)test/test_OutputFileFormat test/input/test_ECAT7OutputFileFormat.in
else
	@echo No ECAT7 support compiled, so no tests for this file format
endif

    
run_interactive_tests_$(dir): all_test_exes
	$(DEST)test/test_interpolate
	$(DEST)test/test_display


include $(WORKSPACE)/test.mk

