#
# $Id$
# 
dir := local/test

$(dir)_TEST_SOURCES :=  \
test_RigidObject3DTransformation.cxx \
	test_Quaternion.cxx

run_tests_$(dir): all_test_exes 
	$(DEST)local/test/test_RigidObject3DTransformation
	$(DEST)local/test/test_Quaternion

 
run_interactive_tests_$(dir): 

# rule to ignore registries
# note: has to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_Fourier: ${DEST}$(dir)/test_Fourier.o $(STIR_LIB) 
	$(CXX) $(CFLAGS)  -o $@ $< $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)

${DEST}$(dir)/test_RigidObject3DTransformation: ${DEST}$(dir)/test_RigidObject3DTransformation.o $(STIR_LIB) 
	$(CXX) $(CFLAGS)  -o $@ $< $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)

include $(WORKSPACE)/test.mk
