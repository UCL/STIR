#
# $Id$
#

dir := local/motion
$(dir)_LIB_SOURCES:= \
	Polaris_MT_File.cxx \
	RegisteredObject.cxx \
	RigidObject3DMotion.cxx\
	RigidObject3DMotionFromPolaris.cxx \
	RigidObject3DTransformation.cxx \
	object_3d_transform_image.cxx

$(dir)_REGISTRY_SOURCES:= local_motion_registries.cxx

include $(WORKSPACE)/lib.mk

