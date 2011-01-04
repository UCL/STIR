#
# $Id$
#
set(dir utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  manip_image 
	manip_projdata 
	display_projdata 
	do_linear_regression 
	postfilter 
	compare_projdata 
	compare_image 
	extract_segments 
	correct_projdata 
	stir_math 
	list_projdata_info 
	list_image_info 
	list_image_values 
	find_maxima_in_image 
	create_projdata_template 
	SSRB 
	poisson_noise 
	get_time_frame_info 
	generate_image 
	list_ROI_values 
	zoom_image 
	find_fwhm_in_image 
	abs_image 
        convert_to_binary_image 
	rebin_projdata 
	write_proj_matrix_by_bin 
	calculate_attenuation_coefficients 
	attenuation_coefficients_to_projections 
)

if (AVW_FOUND)
  list(APPEND ${dir}_EXE_SOURCES  conv_AVW)
endif()

include(../stir_exe_targets.cmake)
