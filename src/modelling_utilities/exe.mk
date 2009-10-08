#
# $Id$
#

dir:=modelling_utilities

$(dir)_SOURCES = \
	apply_patlak_to_images.cxx \
	get_dynamic_images_from_parametric_images.cxx \
	mult_model_with_dyn_images.cxx \
	write_patlak_matrix.cxx \
	mult_image_parameters.cxx 
#read_input_function.cxx



include $(WORKSPACE)/exe.mk
