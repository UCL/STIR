#
# $Id$
# 
dir := recon_test

$(dir)_TEST_SOURCES := test_DataSymmetriesForBins_PET_CartesianGrid.cxx


run_tests_$(dir): all_test_exes 
	$(DEST)recon_test/test_DataSymmetriesForBins_PET_CartesianGrid

    
run_interactive_tests_$(dir): 

# rules that do not link with all registries to save time during linking
# Beware: the pattern below is dangerous as it relies on a naming style for the registries.
# Note: have to be before the include below as that resets $(dir)

${DEST}$(dir)/test_DataSymmetriesForBins_PET_CartesianGrid: ${DEST}$(dir)/test_DataSymmetriesForBins_PET_CartesianGrid.o \
   $(STIR_LIB) $(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES))
	$(CXX) $(CFLAGS)  -o $@ $< \
		$(filter %recon_buildblock_registries.o,$(STIR_REGISTRIES)) \
		 $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)

include $(WORKSPACE)/test.mk


