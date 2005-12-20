#
# $Id$
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
	find_ML_normfactors3D.cxx \
	apply_normfactors3D.cxx \
	create_normfactors3D.cxx \
	find_ML_singles_from_delayed.cxx \
	threshold_norm_data.cxx \
	zero_projdata_from_norm.cxx \
	attenuation_coefficients_to_projections.cxx \
	construct_randoms_from_singles.cxx \
	interpolate_blocks.cxx \
	set_blocks_to_value.cxx \
	shift_projdata_along_axis.cxx \
	remove_sinograms.cxx \
	fit_cylinder.cxx \
	find_maximum_in_image.cxx \
	find_sinogram_rescaling_factors.cxx \
        find_fwhm_in_image.cxx \
	line_profiles_through_projdata.cxx \
	interpolate_projdata.cxx \
	read_input_function.cxx \
	calculate_attenuation_coefficients.cxx \
	make_grid_image.cxx \
	read_dynamic_images.cxx \
	apply_patlak_on_images.cxx \
	apply_patlak_to_images.cxx \
	add_side_shields.cxx 

#	CoG.cxx \
#	make_cylinder.cxx 

ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ecat7 stuff as well
  $(dir)_SOURCES += change_mhead_file_type.cxx copy_ecat7_header.cxx
endif

${DEST}$(dir)/read_input_function: ${DEST}$(dir)/read_input_function$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

include $(WORKSPACE)/exe.mk
