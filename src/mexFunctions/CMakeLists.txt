
option(BUILD_MATLAB_INTERFACE
       "Build Mex functions" OFF)

if (BUILD_MATLAB_INTERFACE)
  include(BuildMex.cmake)

  set(mexFile stir_forward_project)
  BuildMex(MEXNAME ${mexFile} TARGETDIR . SOURCE stir_forward_project.cxx ${STIR_REGISTRIES} LIBRARIES ${STIR_LIBRARIES})

   SET_PROPERTY(TARGET ${mexFile} PROPERTY FOLDER "Matlab")

  install(TARGETS ${mexFile} DESTINATION matlab)
endif()
