#
# $Id$
#

dir:=utilities

$(dir)_SOURCES:=manip_image.cxx \
	manip_projdata.cxx \
	display_projdata.cxx \
	do_linear_regression.cxx \
	postfilter.cxx \
	compare_projdata.cxx \
	compare_image.cxx \
	extract_segments.cxx \
	correct_projdata.cxx \
	stir_math.cxx \
	list_projdata_info.cxx \
	list_image_info.cxx \
	list_image_values.cxx \
	find_maxima_in_image.cxx \
	create_projdata_template.cxx \
	SSRB.cxx \
	poisson_noise.cxx \
	get_time_frame_info.cxx \
	generate_image.cxx \
	list_ROI_values.cxx \
	zoom_image.cxx \
	find_fwhm_in_image.cxx \
	abs_image.cxx \
	rebin_projdata.cxx \
	write_proj_matrix_by_bin.cxx \
	calculate_attenuation_coefficients.cxx \
	attenuation_coefficients_to_projections.cxx 

ifeq ($(HAVE_AVW),1)
  $(dir)_SOURCES += conv_AVW.cxx
endif

${DEST}$(dir)/poisson_noise: ${DEST}$(dir)/poisson_noise$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/conv_AVW: ${DEST}$(dir)/conv_AVW$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


include $(WORKSPACE)/exe.mk
