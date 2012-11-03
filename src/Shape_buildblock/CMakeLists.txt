#
# $Id$
#
set(dir Shape_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  Shape3D 
  DiscretisedShape3D 
  Shape3DWithOrientation 
  Ellipsoid 
  EllipsoidalCylinder 
  Box3D
)

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries

include(../stir_lib_target.cmake)

target_link_libraries(Shape_buildblock buildblock IO )
