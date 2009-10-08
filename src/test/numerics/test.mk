#
# $Id$
# 
dir := test/numerics

$(dir)_TEST_SOURCES := \
	test_matrices.cxx \
	test_overlap_interpolate.cxx \
	test_integrate_discrete_function.cxx




##################################################
# rules to ignore registries
# note: have to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_matrices: ${DEST}$(dir)/test_matrices${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_overlap_interpolate: ${DEST}$(dir)/test_overlap_interpolate${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_integrate_discrete_function: ${DEST}$(dir)/test_integrate_discrete_function${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

include $(WORKSPACE)/test.mk

