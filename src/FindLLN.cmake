
 if (NOT LLN_ROOT_DIR AND NOT $ENV{LLN} STREQUAL "")
    set(LLN_ROOT_DIR $ENV{LLN_ROOT_DIR})
  endif(NOT LLN_ROOT_DIR AND NOT $ENV{LLN} STREQUAL "")

#TODO necessary?
  IF( LLN_ROOT_DIR )
    file(TO_CMAKE_PATH ${LLN_ROOT_DIR} LLN_ROOT_DIR)
  ENDIF( LLN_ROOT_DIR )

  find_path(LLN_INCLUDE_DIRS NAME matrix.h PATHS ${LLN_ROOT_DIR}
        DOC "location of LLN include files")
#  message(STATUS "LLN_INCLUDE_DIRS ${LLN_INCLUDE_DIRS}")
  
  find_library(LLN_LIBRARIES NAME ecat HINTS ${LLN_ROOT_DIR}
        DOC "location of LLN library")

  if(LLN_LIBRARIES AND $(CMAKE_SYSTEM_NAME) STREQUAL "SunOS")
       find_library(LLN_EXTRA_LIBS NAME nsl socket
        DOC "extra libraries for linking with the LLN matrix library")
       target_link_libraries(${LLN_LIBRARIES} ${LLN_EXTRA_LIBS})
  endif()

# handle the QUIETLY and REQUIRED arguments and set LLN_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LLN "LLN matrix library not found. If you do have it, set the missing variables" LLN_LIBRARIES LLN_INCLUDE_DIRS)

