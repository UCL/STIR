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
	MatchTrackerAndScanner.cxx \
	transform_3d_object.cxx

$(dir)_REGISTRY_SOURCES:= local_motion_registries.cxx

include $(WORKSPACE)/lib.mk

