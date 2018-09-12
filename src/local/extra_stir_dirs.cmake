# add include directory to compiler switches
include_directories(${STIR_LOCAL}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/local/include)

# add registries
list(APPEND STIR_REGISTRIES 
#  ${STIR_LOCAL}/listmode_buildblock/UCL_listmode_registries 
  ${CMAKE_SOURCE_DIR}/src/local/buildblock/local_buildblock_registries
  ${CMAKE_SOURCE_DIR}/src/local/recon_buildblock/local_recon_buildblock_registries)

# add to list of libraries to include in linking
#list(APPEND STIR_LIBRARIES UCL_listmode_buildblock) 
list(APPEND STIR_LIBRARIES local_buildblock local_motion_buildblock)
list(APPEND STIR_LIBRARIES local_recon_buildblock) 
#list(APPEND STIR_LIBRARIES local_listmode_buildblock)

# check CMakeLists in next directories
#add_subdirectory( ${STIR_LOCAL}/listmode_buildblock local/listmode_buildblock)

#add_subdirectory( ${STIR_LOCAL}/listmode_utilities local/listmode_utilities)

add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/buildblock local/buildblock)
add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/recon_buildblock local/GE_recon_buildblock)
add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/motion local/motion_buildblock)
add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/motion_utilities local/motion_utilities)
#add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/listmode local/listmode_buildblock_HI)
#add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/listmode_utilities local/listmode_utilities_HI)
#add_subdirectory( ${CMAKE_SOURCE_DIR}/src/local/utilities local/utilities_HI)