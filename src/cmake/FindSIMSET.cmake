# In order to link with SimSET either build it as static
# or go to the obj path and issue the command ar rcs simset.a ./*.o
# and then link to that. 

 if (NOT SIMSET_ROOT_DIR AND NOT $ENV{SIMSET} STREQUAL "")
    set(SIMSET_ROOT_DIR $ENV{SIMSET_ROOT_DIR})
  endif(NOT SIMSET_ROOT_DIR AND NOT $ENV{SIMSET} STREQUAL "")

#TODO necessary?
  IF( SIMSET_ROOT_DIR )
    file(TO_CMAKE_PATH ${SIMSET_ROOT_DIR} SIMSET_ROOT_DIR)
  ENDIF( SIMSET_ROOT_DIR )

  find_path(SIMSET_INCLUDE_DIRS NAME SystemDependent.h PATHS ${SIMSET_ROOT_DIR}
        DOC "location of SIMSET include files")
  
  find_library(SIMSET_LIBRARY NAME simset HINTS ${SIMSET_ROOT_DIR}
        DOC "location of SIMSET library")
        
if (SIMSET_LIBRARY)
  message(STATUS "AVAILABLE SIMSET LIBRARY: " ${SIMSET_LIBRARY})
endif()

# handle the QUIETLY and REQUIRED arguments and set LLN_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SIMSET "SIMSET library not found. If you do have it, set the missing variables" SIMSET_LIBRARY SIMSET_INCLUDE_DIRS)

