#
# $Id$
# 
dir := test

$(dir)_TEST_SOURCES := test_Array.cxx test_NestedIterator.cxx \
	test_VectorWithOffset.cxx \
	test_ArrayFilter.cxx \
        test_convert_array.cxx \
	test_IndexRange.cxx \
	test_coordinates.cxx \
	test_linear_regression.cxx \
	test_filename_functions.cxx \
	test_coordinates.cxx \
	test_VoxelsOnCartesianGrid.cxx \
	test_zoom_image.cxx \
	test_proj_data_info.cxx \
	test_stir_math.cxx \
	test_OutputFileFormat.cxx \
	test_ByteOrder.cxx \
	test_Scanner.cxx \
	test_ROIs.cxx \
	test_VAXfloat.cxx \
	test_ArcCorrection.cxx

$(dir)_INTERACTIVE_TEST_SOURCES := \
	test_display.cxx \
	test_interpolate.cxx 


# note: do not use $(dir) in the command lines as that variable
# will be something else when the command line gets executed! 
# (see e.g. lib.mk)
run_$(dir)/test_linear_regression: $(DEST)$(dir)/test_linear_regression PHONY_TARGET
	$<  test/input/test_linear_regression.in

run_$(dir)/test_OutputFileFormat: $(DEST)$(dir)/test_OutputFileFormat PHONY_TARGET
	$< test/input/test_InterfileOutputFileFormat.in 
	$< test/input/test_InterfileOutputFileFormat_short.in 
	$< test/input/test_ECAT6OutputFileFormat.in
ifeq ($(HAVE_LLN_MATRIX),1)
	$< test/input/test_ECAT7OutputFileFormat.in
else
	@echo No ECAT7 support compiled, so no tests for this file format
endif


ifeq ("$(IS_MS_VC)","")
run_$(dir)/test_stir_math: $(DEST)$(dir)/test_stir_math $(DEST)utilities/stir_math PHONY_TARGET
	$< $(DEST)utilities/stir_math$(EXE_SUFFIX) 
else
run_$(dir)/test_stir_math: $(DEST)$(dir)/test_stir_math $(DEST)utilities/stir_math PHONY_TARGET
	$< `cygpath -w $(DEST)utilities/stir_math.exe`
endif

run_$(dir)/test_VAXfloat:  $(DEST)$(dir)/test_VAXfloat PHONY_TARGET
	$< --read test/input/test_VAXfloat.in

##################################################
# rules to ignore registries
# note: have to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_ArrayFilter: ${DEST}$(dir)/test_ArrayFilter${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

ifeq ("$(FAST_test)","")
${DEST}$(dir)/test_VectorWithOffset: ${DEST}$(dir)/test_VectorWithOffset${O_SUFFIX} $(STIR_LIB)
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_Array: ${DEST}$(dir)/test_Array${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_convert_array: ${DEST}$(dir)/test_convert_array${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


else
  # rules for test_VectorWithOffset et al that ignores $(STIR_LIB) completely as it's all inline (except for error())
  # somewhat dangerous though in case files/rules change
${DEST}$(dir)/test_VectorWithOffset: ${DEST}$(dir)/test_VectorWithOffset${O_SUFFIX} \
    ${DEST}buildblock/error${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $<  ${DEST}buildblock/error${O_SUFFIX} $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_Array: ${DEST}$(dir)/test_Array${O_SUFFIX} \
   ${DEST}buildblock/error${O_SUFFIX} ${DEST}buildblock/warning${O_SUFFIX} \
   $(DEST)buildblock/IndexRange${O_SUFFIX} ${DEST}buildblock/ByteOrder${O_SUFFIX}  \
   $(DEST)buildblock/utilities${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< \
	${DEST}buildblock/error${O_SUFFIX}  $(DEST)buildblock/IndexRange${O_SUFFIX} \
	${DEST}buildblock/warning${O_SUFFIX} ${DEST}buildblock/ByteOrder${O_SUFFIX}  \
	$(DEST)buildblock/utilities${O_SUFFIX} \
	$(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_NestedIterator: ${DEST}$(dir)/test_NestedIterator${O_SUFFIX} ${DEST}buildblock/error${O_SUFFIX}  $(DEST)buildblock/IndexRange${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $<  ${DEST}buildblock/error${O_SUFFIX}  $(DEST)buildblock/IndexRange${O_SUFFIX} $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_convert_array: ${DEST}$(dir)/test_convert_array${O_SUFFIX} \
   ${DEST}buildblock/error${O_SUFFIX} ${DEST}buildblock/warning${O_SUFFIX} 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< \
	${DEST}buildblock/error${O_SUFFIX}  \
	${DEST}buildblock/warning${O_SUFFIX}   \
$(LINKFLAGS) $(SYS_LIBS)

endif



${DEST}$(dir)/test_IndexRange: ${DEST}$(dir)/test_IndexRange${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


${DEST}$(dir)/test_coordinates: ${DEST}$(dir)/test_coordinates${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


${DEST}$(dir)/test_linear_regression: ${DEST}$(dir)/test_linear_regression${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


${DEST}$(dir)/test_interpolate: ${DEST}$(dir)/test_interpolate${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


${DEST}$(dir)/test_filename_functions: ${DEST}$(dir)/test_filename_functions${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_ByteOrder: ${DEST}$(dir)/test_ByteOrder${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_stir_math: ${DEST}$(dir)/test_stir_math${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_OutputFileFormat: ${DEST}$(dir)/test_OutputFileFormat${O_SUFFIX} \
   $(STIR_LIB) $(filter %IO_registries${O_SUFFIX},$(STIR_REGISTRIES))  $(EXTRA_LIBS) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< \
		$(filter %IO_registries${O_SUFFIX},$(STIR_REGISTRIES)) \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_proj_data_info: ${DEST}$(dir)/test_proj_data_info${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_ArcCorrection: ${DEST}$(dir)/test_ArcCorrection${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


include $(WORKSPACE)/test.mk

