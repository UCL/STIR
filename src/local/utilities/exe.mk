#
#

dir:=local/utilities

$(dir)_SOURCES = \
	fillwith1.cxx \
	fillwithotherprojdata.cxx \
	compute_plane_rescale_factors.cxx \
	apply_plane_rescale_factors.cxx \
	prepare_projdata.cxx \
	normalizedbckproj.cxx \
	inverse_proj_data.cxx 	\
	threshold_norm_data.cxx \
	create_normfactors3D.cxx \
	zero_projdata_from_norm.cxx \
	line_profiles_through_projdata.cxx \
	interpolate_projdata.cxx \
	image_flip_x.cxx \
	Hounsfield2mu.cxx

#	list_TAC_ROI_values.cxx \
#	make_grid_image.cxx \
#	Bland_Altman_plot.cxx \
#	get_total_counts.cxx \
#	precompute_denominator_SPS.cxx \
#	interpolate_blocks.cxx \
#	set_blocks_to_value.cxx \
#	shift_projdata_along_axis.cxx \
#	remove_sinograms.cxx \
#	fit_cylinder.cxx \
#	find_sinogram_rescaling_factors.cxx \
#	CoG.cxx \
#	add_time_frame_info.cxx \
#	make_cylinder.cxx 

ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ecat7 stuff as well
  $(dir)_SOURCES += change_mhead_file_type.cxx 
endif

ifeq ($(HAVE_RDF),1)
  $(dir)_SOURCES += print_rdf_singles.cxx
endif

ifeq ($(HAVE_AVW),1)
  $(dir)_SOURCES += AVWROI.cxx
endif


${DEST}$(dir)/read_input_function: ${DEST}$(dir)/read_input_function$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/AVWROI: ${DEST}$(dir)/AVWROI$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

include $(WORKSPACE)/exe.mk
