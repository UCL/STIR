#
#
dir := Shape_buildblock

$(dir)_LIB_SOURCES = \
  Shape3D.cxx \
  DiscretisedShape3D.cxx \
  Shape3DWithOrientation.cxx \
  Ellipsoid.cxx \
  EllipsoidalCylinder.cxx \
  Box3D.cxx


$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk

