#
# $Id$
# 
dir := local/test/test

$(dir)_TEST_SOURCES := 
test_RigidObject3DTransformation.cxx \
	test_Quaternion.cxx

run_tests_$(dir): all_test_exes 
	$(DEST)local/test/test_RigidObject3DTransformation
	$(DEST)local/test/test_Quaternion

    
run_interactive_tests_$(dir): 


include $(WORKSPACE)/test.mk

