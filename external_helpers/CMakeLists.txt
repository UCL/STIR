# include all git submodules that should be built with STIR

#### Check if std::format exists (and is good enough). If not, use fmt submodule.

set(CODE "
#include <version>
#if defined(__cpp_lib_format) && (__cpp_lib_format >= 201907L)
#  include <format>
namespace internal_format = std; // using std::format;
#else
#error std::format not present or not good enough
#endif
int main() { return 0; }
")

check_cxx_source_compiles("${CODE}" USE_STD_FORMAT)

if(USE_STD_FORMAT)

    message(STATUS "Will use std::format.")

else()

    message(STATUS "std::format missing. Will use our fmt submodule.")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external_helpers/fmt/CMakeLists.txt")
      message(FATAL_ERROR "The {fmt} submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
    endif()

  add_subdirectory(fmt)

  # ensure fmt is built with position independent code
  if (TARGET fmt)
    set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()

  # In fact, stir/format.h will set it to use fmt header-only, so we only want to know
  # where the headers are.
  get_target_property(FMT_INCLUDE_DIRS fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
  set(FMT_INCLUDE_DIRS ${FMT_INCLUDE_DIRS} PARENT_SCOPE)
endif()

