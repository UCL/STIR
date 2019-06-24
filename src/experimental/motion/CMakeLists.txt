set(dir local_motion_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	RigidObject3DTransformation.cxx
        Transform3DObjectImageProcessor.cxx
        NonRigidObjectTransformationUsingBSplines.cxx
	transform_3d_object.cxx
)

include(stir_lib_target)

target_link_libraries(${dir} numerics_buildblock local_buildblock )

