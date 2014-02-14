#
# 
dir := test/modelling

$(dir)_TEST_SOURCES :=  \
	test_modelling.cxx \
	test_ParametricDiscretisedDensity.cxx 

run_$(dir)/test_modelling: $(DEST)$(dir)/test_modelling PHONY_TARGET
	$< test/modelling/input/

include $(WORKSPACE)/test.mk
