#
# $Id$
# 
dir := local/recon_test

$(dir)_TEST_SOURCES :=  \
test_ProjMatrixByBinUsingInterpolation.cxx


run_tests_$(dir): all_test_exes 
	$(DEST)local/recon_test/test_ProjMatrixByBinUsingInterpolation

 
run_interactive_tests_$(dir): 

# rule to ignore registries
# note: has to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_ProjMatrixByBinUsingInterpolation: \
   ${DEST}$(dir)/test_ProjMatrixByBinUsingInterpolation.o $(STIR_LIB) \
   $(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES))
	$(CXX) $(CFLAGS)  -o $@ $< \
		$(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES)) \
		$(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)

include $(WORKSPACE)/test.mk
