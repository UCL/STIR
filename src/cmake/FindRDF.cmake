
 if (NOT RDF_ROOT_DIR AND NOT $ENV{RDF} STREQUAL "")
    set(RDF_ROOT_DIR $ENV{RDF_ROOT_DIR})
  endif(NOT RDF_ROOT_DIR AND NOT $ENV{RDF} STREQUAL "")

#TODO necessary?
  IF( RDF_ROOT_DIR )
    file(TO_CMAKE_PATH ${RDF_ROOT_DIR} RDF_ROOT_DIR)
  ENDIF( RDF_ROOT_DIR )

  find_path(RDF_INCLUDE_DIRS NAME GErdfUtils.h PATHS ${RDF_ROOT_DIR}
        DOC "location of RDF include files")
#  message(STATUS "RDF_INCLUDE_DIRS ${RDF_INCLUDE_DIRS}")
  
  find_library(RDF_LIBRARIES NAME GEio HINTS ${RDF_ROOT_DIR}
        DOC "location of RDF library")

# handle the QUIETLY and REQUIRED arguments and set RDF_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(RDF "GE RDF IO library not found. If you do have it, set the missing variables" RDF_LIBRARIES RDF_INCLUDE_DIRS)

