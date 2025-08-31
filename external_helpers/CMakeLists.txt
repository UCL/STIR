# include all git submodules that should be built with STIR

#### Check if std::format exists (and is good enough)

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
    # Use this variable to add a dependency for every library/utility that uses format()
    set(FMT_DEPENDENCY PRIVATE fmt::fmt-header-only PARENT_SCOPE)

  add_subdirectory(fmt)

  # ensure fmt is built with position independent code
  if (TARGET fmt)
    set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endif()

