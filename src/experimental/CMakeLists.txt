# add registries
list(APPEND STIR_REGISTRIES 
#  ${STIR_LOCAL}/listmode_buildblock/UCL_listmode_registries 
  ${CMAKE_CURRENT_SOURCE_DIR}/buildblock/local_buildblock_registries
  ${CMAKE_CURRENT_SOURCE_DIR}/recon_buildblock/local_recon_buildblock_registries)

# add to list of libraries to include in linking
#list(APPEND STIR_LIBRARIES UCL_listmode_buildblock) 
list(APPEND STIR_LIBRARIES local_buildblock local_motion_buildblock)
list(APPEND STIR_LIBRARIES local_recon_buildblock) 
#list(APPEND STIR_LIBRARIES local_listmode_buildblock)

# check CMakeLists in next directories
#add_subdirectory( ${STIR_LOCAL}/listmode_buildblock listmode_buildblock)

#add_subdirectory( ${STIR_LOCAL}/listmode_utilities listmode_utilities)

add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/buildblock buildblock)
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/recon_buildblock GE_recon_buildblock)
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/motion motion_buildblock)
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/motion_utilities motion_utilities)
#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/listmode listmode_buildblock_HI)
#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/listmode_utilities listmode_utilities_HI)
#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/utilities utilities_HI)

add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/motion_test motion_test)