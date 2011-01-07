# expects files in $AVW/include and $AVW/$TARGET/lib

 set(AVW_ROOT_DIR CACHE PATH "root of AVW, normally found from the AVW environment variable")
 if (NOT AVW_ROOT_DIR AND NOT $ENV{AVW} STREQUAL "")
    set(AVW_ROOT_DIR $ENV{AVW_ROOT_DIR})
  endif(NOT AVW_ROOT_DIR AND NOT $ENV{AVW} STREQUAL "")

#TODO necessary?
  IF( AVW_ROOT_DIR )
    file(TO_CMAKE_PATH ${AVW_ROOT_DIR} AVW_ROOT_DIR)
  ENDIF( AVW_ROOT_DIR )

  find_path(AVW_INCLUDE_DIRS NAME AVW.h PATHS ${AVW_ROOT_DIR}/include
        DOC "location of AVW include files")
#  message(STATUS "AVW_INCLUDE_DIRS ${AVW_INCLUDE_DIRS}")
  
  find_library(AVW_LIBRARIES NAME AVW
        HINTS ${AVW_ROOT_DIR}/$ENV(TARGET)/lib/
        DOC "location of AVW library")

# handle the QUIETLY and REQUIRED arguments and set AVW_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(AVW "AVW matrix library not found. If you do have it, set the missing variables" AVW_LIBRARIES AVW_INCLUDE_DIRS)

MARK_AS_ADVANCED(AVW_LIBRARIES AVW_INCLUDE_DIRS )