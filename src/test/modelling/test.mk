#
# $Id$
# 
dir := test/modelling

$(dir)_TEST_SOURCES :=  \
	test_modelling.cxx \
	test_ParametricDiscretisedDensity.cxx 


include $(WORKSPACE)/test.mk
