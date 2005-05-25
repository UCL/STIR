#
# $Id$
# 
dir := local/test

$(dir)_TEST_SOURCES :=  \
test_RigidObject3DTransformation.cxx \
	test_Quaternion.cxx \
	test_proj_data_info_LOR.cxx \
	test_IR_filters.cxx \
	test_BSplines.cxx        

# rule to ignore registries
# note: has to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_Fourier: ${DEST}$(dir)/test_Fourier$(O_SUFFIX) $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

# need to add this as it's not in the TEST_SOURCES macro
ifneq ($(MAKECMDGOALS:clean%=clean),clean)
 -include ${DEST}$(dir)/test_Fourier.P
endif

run_$(dir)/test_proj_data_info_LOR: $(DEST)$(dir)/test_proj_data_info_LOR PHONY_TARGET
	$<  4 5 60 7 8 40 10 45 12 5 3 60

${DEST}$(dir)/test_RigidObject3DTransformation: ${DEST}$(dir)/test_RigidObject3DTransformation$(O_SUFFIX) $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

include $(WORKSPACE)/test.mk
