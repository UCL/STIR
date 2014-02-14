#
# 
dir := recon_test

$(dir)_TEST_SOURCES := test_DataSymmetriesForBins_PET_CartesianGrid.cxx \
  test_PoissonLogLikelihoodWithLinearModelForMeanAndProjData.cxx


# rules that do not link with all registries to save time during linking
# Beware: the pattern below is dangerous as it relies on a naming style for the registries.
# Note: have to be before the include below as that resets $(dir)

${DEST}$(dir)/test_DataSymmetriesForBins_PET_CartesianGrid: ${DEST}$(dir)/test_DataSymmetriesForBins_PET_CartesianGrid${O_SUFFIX} \
   $(STIR_LIB) $(filter %recon_buildblock_registries${O_SUFFIX},$(STIR_REGISTRIES))
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		$(filter %recon_buildblock_registries${O_SUFFIX},$(STIR_REGISTRIES)) \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

.PHONY: run_$(dir)/test_modelling.sh

run_$(dir)/test_modelling.sh: \
  $(dir)/test_modelling.sh \
  $(DEST)/iterative/POSMAPOSL/PatlakOSMAPOSL \
  $(DEST)/utilities/get_dynamic_images_from_parametric_images \
  $(DEST)/utilities/mult_model_with_dyn_images \
  $(DEST)/utilities/apply_patlak_to_images  \
  $(DEST)/utilities/stir_math \
  $(DEST)/utilities/generate_image \
  $(DEST)/utilities/ifheaders_for_ecat7 \
  $(DEST)/utilities/compare_image \
  $(DEST)/utilities/manip_image \
  $(DEST)/utilities/conv_to_ecat7 \
  $(DEST)/recon_test/fwdtest
	DEST=$(DEST) bash $<

include $(WORKSPACE)/test.mk
