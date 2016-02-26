#
#

dir:=scripts

$(dir)_SCRIPTS:=stir_subtract stir_divide count stir_print_voxel_sizes.sh \
  estimate_scatter.sh \
  zoom_att_image.sh \
  get_num_voxels.sh


include $(WORKSPACE)/exe.mk
