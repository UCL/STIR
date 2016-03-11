find_program(HAS_GLOBAL_ROOT "root-config")

if (HAS_GLOBAL_ROOT)

  set (root_cmd "root-config")
  set (root_incl_arg "--incdir")
  set (root_lib_arg "--libs")

  execute_process(COMMAND ${root_cmd} ${root_incl_arg} OUTPUT_VARIABLE
  CERN_ROOT_INCLUDE_DIRS)

  execute_process(COMMAND ${root_cmd} ${root_lib_arg} OUTPUT_VARIABLE
  TCERN_ROOT_LIBRARIES)

  string (REPLACE ";" " " TCERN_ROOT_LIBRARIES "${CERN_ROOT_LIBRARIES}")

endif(HAS_GLOBAL_ROOT)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CERN_ROOT "CERN ROOT not found. If you do have it, set the missing variables" CERN_ROOT_LIBRARIES CERN_ROOT_INCLUDE_DIRS)