#
# $Id$
# 
dir := recon_test

$(dir)_TEST_SOURCES := test_DataSymmetriesForBins_PET_CartesianGrid.cxx


run_tests_$(dir): all_test_exes 
	$(DEST)recon_test/test_DataSymmetriesForBins_PET_CartesianGrid

    
run_interactive_tests_$(dir): 

include $(WORKSPACE)/test.mk


