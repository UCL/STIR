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
  \brief Include for formatting function. If standard library version available (C++20 and newer),
         use it. Otherwise use fmt::format, which is included into STIR as a git submodule.

  \author Markus Jehl
  \author Kris Thielemans
*/

#if __has_include(<format>)
#  include <format>
using std::format;
#else
#  include <fmt/format.h>
using fmt::format;
#endif

#endif