#
# $Id$
# 
dir := local/test

$(dir)_TEST_SOURCES :=  \
	test_Quaternion.cxx \
	test_proj_data_info_LOR.cxx \
	test_erf.cxx 

ifeq ($(STIR_DEVEL_MOTION),1)
$(dir)_TEST_SOURCES +=  \
  test_RigidObject3DTransformation.cxx
endif


# rule to ignore registries
# note: has to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_Fourier: ${DEST}$(dir)/test_Fourier$(O_SUFFIX) $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

# need to add this as it's not in the TEST_SOURCES macro
ifneq ($(MAKECMDGOALS:clean%=clean),clean)
 -include ${DEST}$(dir)/test_Fourier.P
endif

run_$(dir)/test_proj_data_info_LOR: $(DEST)$(dir)/test_proj_data_info_LOR PHONY_TARGET
	$<  4 5 60 7 8 40 10 45 12 5 3 60

run_$(dir)/test_IO_ParametricDiscretisedDensity: $(DEST)$(dir)/test_IO_ParametricDiscretisedDensity PHONY_TARGET
	$< STIRtmp_dyn2f.img STIRtmp_dyn2f_out.img 


${DEST}$(dir)/test_RigidObject3DTransformation: ${DEST}$(dir)/test_RigidObject3DTransformation$(O_SUFFIX) $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

.PHONY: run_$(dir)/test_nonseparable_convolution_with_image_kernel.sh

run_$(dir)/test_nonseparable_convolution_with_image_kernel.sh: \
  $(dir)/test_nonseparable_convolution_with_image_kernel.sh \
  $(DEST)/local/utilities/create_a_point  \
  $(DEST)/utilities/generate_image \
  $(DEST)/utilities/compare_image \
  $(DEST)/utilities/postfilter 
	DEST=$(DEST) bash $<

include $(WORKSPACE)/test.mk
