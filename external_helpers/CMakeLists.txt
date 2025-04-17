# include all git submodules that should be built with STIR
add_subdirectory(fmt)

# ensure fmt is built with position independent code
if (TARGET fmt)
  set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()