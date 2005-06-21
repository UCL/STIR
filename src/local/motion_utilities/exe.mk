#
# $Id$
#

dir:=local/motion_utilities

$(dir)_SOURCES = \
	rigid_object_transform_projdata.cxx \
	rigid_object_transform_image.cxx \
	move_image.cxx \
	move_projdata.cxx \
	remove_corrupted_sinograms.cxx \
	fwd_image_and_fill_missing_data.cxx \
	add_planes_to_image.cxx \
	sync_polaris.cxx \
	find_motion_corrected_norm_factors.cxx \
	match_tracker_and_scanner.cxx

include $(WORKSPACE)/exe.mk
