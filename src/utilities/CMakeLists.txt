#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  manip_image.cxx
  stir_write_pgm.cxx
	manip_projdata.cxx
	display_projdata.cxx
	do_linear_regression.cxx
	postfilter.cxx
	compare_projdata.cxx
	compare_image.cxx
	extract_segments.cxx
	extract_single_images_from_dynamic_image.cxx
        estimate_triple_energy_window_scatter_sinogram.cxx
	correct_projdata.cxx
	stir_math.cxx
	list_projdata_info.cxx
	list_image_info.cxx
	list_image_values.cxx
        list_detector_and_bin_info.cxx
	find_maxima_in_image.cxx
        create_multi_header.cxx
	create_projdata_template.cxx
	SSRB.cxx
        invert_axis.cxx
	poisson_noise.cxx
	get_time_frame_info.cxx
	generate_image.cxx
	list_ROI_values.cxx
	zoom_image.cxx
	find_fwhm_in_image.cxx
	abs_image.cxx
        convert_to_binary_image.cxx
	rebin_projdata.cxx
	write_proj_matrix_by_bin.cxx
        forward_project.cxx
        back_project.cxx
	calculate_attenuation_coefficients.cxx
	compute_sqrt_Hessian_row_sum.cxx
	attenuation_coefficients_to_projections.cxx
        conv_GATE_raw_ECAT_projdata_to_interfile.cxx
        conv_gipl_to_interfile.cxx
        conv_interfile_to_gipl.cxx
        shift_image.cxx
        shift_image_origin.cxx
        warp_and_accumulate_gated_images.cxx
        warp_image.cxx
        zeropad_planes.cxx
	apply_normfactors3D.cxx
	apply_normfactors.cxx
	construct_randoms_from_singles.cxx
	find_ML_normfactors3D.cxx
	find_ML_normfactors.cxx
	find_ML_singles_from_delayed.cxx
)

if (AVW_FOUND)
  list(APPEND ${dir_EXE_SOURCES}  conv_AVW.cxx)
endif()

if (HDF5_FOUND)
 list(APPEND ${dir_EXE_SOURCES}
   construct_randoms_from_GEsingles.cxx
   GE/print_GE_singles_values.cxx
 )
endif()

if (nlohmann_json_FOUND)
  list(APPEND ${dir_EXE_SOURCES}  ctac_to_mu_values.cxx)

  install(DIRECTORY share/ DESTINATION ${STIR_DATA_DIR}
		  FILES_MATCHING PATTERN "*.json")
endif()

include(stir_exe_targets)
