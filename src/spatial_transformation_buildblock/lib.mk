#
# $Id$
#

dir := spatial_transformation_buildblock
$(dir)_LIB_SOURCES:= SpatialTransformation.cxx \
		     GatedSpatialTransformation.cxx \
                     warp_image.cxx

$(dir)_REGISTRY_SOURCES:= spatial_transformation_registries.cxx

include $(WORKSPACE)/lib.mk
