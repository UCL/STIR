#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.
set(dir utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
  manip_image 
  stir_write_pgm
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
        list_detector_and_bin_info
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
        conv_GATE_raw_ECAT_projdata_to_interfile
        conv_gipl_to_interfile
        conv_interfile_to_gipl
        shift_image
        shift_image_origin
        warp_and_accumulate_gated_images
        warp_image
        zeropad_planes
)

if (AVW_FOUND)
  list(APPEND ${dir_EXE_SOURCES}  conv_AVW)
endif()

include(../stir_exe_targets.cmake)
