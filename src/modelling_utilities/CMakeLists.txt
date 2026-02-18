#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file for building STIR.

set(dir modelling_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	apply_patlak_to_images.cxx
	get_dynamic_images_from_parametric_images.cxx
	make_parametric_image_from_components.cxx
	extract_single_images_from_parametric_image.cxx
	mult_model_with_dyn_images.cxx
	write_patlak_matrix.cxx
	mult_image_parameters.cxx
)

include(stir_exe_targets)
