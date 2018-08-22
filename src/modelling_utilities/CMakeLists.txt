#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans

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

# cmake helper file for building STIR.

set(dir modelling_utilities)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
	apply_patlak_to_images
	get_dynamic_images_from_parametric_images
	make_parametric_image_from_components
	split_dynamic_images
	mult_model_with_dyn_images
	write_patlak_matrix
	mult_image_parameters
)

include(stir_exe_targets)
