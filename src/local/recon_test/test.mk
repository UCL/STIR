#
# $Id$
# 
dir := local/recon_test

$(dir)_TEST_SOURCES :=  \
test_ProjMatrixByBinUsingInterpolation.cxx

# rule to ignore registries
# note: has to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_ProjMatrixByBinUsingInterpolation: \
   ${DEST}$(dir)/test_ProjMatrixByBinUsingInterpolation$(O_SUFFIX) $(STIR_LIB) \
   $(filter %recon_buildblock_registries$(O_SUFFIX),$(STIR_REGISTRIES))
	$(CXX) $(CFLAGS)  $(EXE_OUTFLAG)$@ $< \
		$(filter %recon_buildblock_registries$(O_SUFFIX),$(STIR_REGISTRIES)) \
		$(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

.PHONY: run_$(dir)/test_modelling.sh

run_$(dir)/test_modelling.sh: \
  $(dir)/test_modelling.sh \
  $(dir)/../scripts/copy_frame_info.sh \
  $(dir)/../scripts/min_counts_in_images.sh \
  $(dir)/../scripts/max_counts_in_images.sh \
  $(dir)/../scripts/extract_frames.sh \
  $(DEST)/local/iterative/POSMAPOSL/PatlakOSMAPOSL \
  $(DEST)/local/utilities/get_dynamic_images_from_parametric_images \
  $(DEST)/local/utilities/mult_model_with_dyn_images \
  $(DEST)/local/utilities/apply_patlak_to_images  \
  $(DEST)/utilities/stir_math \
  $(DEST)/utilities/generate_image \
  $(DEST)/utilities/ifheaders_for_ecat7 \
  $(DEST)/utilities/compare_image \
  $(DEST)/utilities/conv_to_ecat7 \
  $(DEST)/recon_test/fwdtest
	DEST=$(DEST) bash $<

include $(WORKSPACE)/test.mk
