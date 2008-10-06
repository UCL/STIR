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
	create_projdata_template.cxx \
	SSRB.cxx \
	poisson_noise.cxx \
	get_time_frame_info.cxx \
	generate_image.cxx \
	list_ROI_values.cxx \
	zoom_image.cxx \
  	rebin_projdata.cxx \
	write_proj_matrix_by_bin.cxx

ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ecat utilities as well
  $(dir)_SOURCES += ifheaders_for_ecat7.cxx conv_to_ecat7.cxx print_ecat_singles_values.cxx \
	convecat6_if.cxx \
        conv_to_ecat6.cxx \
	ecat_swap_corners.cxx 
endif

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
