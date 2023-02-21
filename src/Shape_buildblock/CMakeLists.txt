#
#
set(dir Shape_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  Shape3D.cxx
  DiscretisedShape3D.cxx
  Shape3DWithOrientation.cxx
  Ellipsoid.cxx
  EllipsoidalCylinder.cxx
  Box3D.cxx
  GenerateImage.cxx
)

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries

include(stir_lib_target)

target_link_libraries(Shape_buildblock buildblock IO numerics_buildblock )
