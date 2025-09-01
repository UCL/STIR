/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_FORMAT__
#define __stir_FORMAT__

/*!
  \file
  \brief Include for formatting function. Use standard library version if available (C++20 and newer).
  Otherwise use fmt::format, which is included into STIR as a git submodule.

  \author Markus Jehl
  \author Kris Thielemans
*/

#include <string>
#include <utility> // for std::forward
#include "stir/common.h"

#if __cplusplus >= 202002L // C++-20
#  include <version>       // need this for __cpp_lib_format
#endif

#if defined(__cpp_lib_format) && (__cpp_lib_format >= 201907L)
#  include <format>
namespace internal_format = std; // using std::format;
#else
#  define FMT_HEADER_ONLY 1
#  if defined(_MSC_VER)
// MS VC compiler needs the /utf-8 compiler switch to be able to handle fmt with Unicode support
// Instead, we just disable it.
#    define FMT_UNICODE 0
#  endif
#  include "fmt/format.h"
namespace internal_format = fmt; // using fmt::format;
#endif

START_NAMESPACE_STIR

#if defined(__cpp_lib_format) && (__cpp_lib_format >= 201907L)
using std::format;
#else

template <typename... Args>
std::string
format(fmt::format_string<Args...> fmt, Args&&... args)
{
  return fmt::format(fmt, std::forward<Args>(args)...);
}
#endif

// this is an alternative function definition for instances where the format string is not known at compile time
// likely will be in c++26
template <typename... Args>
std::string
runtime_format(internal_format::string_view fmt, Args&&... args)
{
#if defined(__cpp_lib_format) && (__cpp_lib_format >= 201907L)
  return internal_format::vformat(fmt, std::make_format_args(args...));
#else
  return internal_format::format(fmt::runtime(fmt), std::forward<Args>(args)...);
#endif
}

END_NAMESPACE_STIR

#endif // __stir_FORMAT__
