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

include $(WORKSPACE)/test.mk
