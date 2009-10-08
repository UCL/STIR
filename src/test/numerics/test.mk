#
# $Id$
# 
dir := test/numerics

$(dir)_TEST_SOURCES := \
	test_matrices.cxx \
	test_overlap_interpolate.cxx \
	test_integrate_discrete_function.cxx \
	test_IR_filters.cxx \
	test_BSplines.cxx \
	test_BSplinesRegularGrid1D.cxx \
	test_BSplinesRegularGrid.cxx 




##################################################
# rules to ignore registries
# note: have to be before include statement as that changes value of $(dir)
${DEST}$(dir)/test_matrices: ${DEST}$(dir)/test_matrices${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_overlap_interpolate: ${DEST}$(dir)/test_overlap_interpolate${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_integrate_discrete_function: ${DEST}$(dir)/test_integrate_discrete_function${O_SUFFIX} $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

ifeq ("$(FAST_test)","")

${DEST}$(dir)/test_BSplinesRegularGrid: ${DEST}$(dir)/test_BSplinesRegularGrid$(O_SUFFIX) $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $< $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

else

${DEST}$(dir)/test_BSplinesRegularGrid: ${DEST}$(dir)/test_BSplinesRegularGrid${O_SUFFIX} \
    ${DEST}buildblock/error${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $<  ${DEST}buildblock/error${O_SUFFIX} $(LINKFLAGS) $(SYS_LIBS)


${DEST}$(dir)/test_BSplinesRegularGrid1D: ${DEST}$(dir)/test_BSplinesRegularGrid1D${O_SUFFIX} \
    ${DEST}buildblock/error${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $<  ${DEST}buildblock/error${O_SUFFIX} $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_BSplines: ${DEST}$(dir)/test_BSplines${O_SUFFIX} \
    ${DEST}buildblock/error${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $<  ${DEST}buildblock/error${O_SUFFIX} $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/test_IR_filters: ${DEST}$(dir)/test_IR_filters${O_SUFFIX} \
    ${DEST}buildblock/error${O_SUFFIX}
	$(LINK) $(EXE_OUTFLAG)$(@)$(EXE_SUFFIX) $<  ${DEST}buildblock/error${O_SUFFIX} $(LINKFLAGS) $(SYS_LIBS)

endif

include $(WORKSPACE)/test.mk

