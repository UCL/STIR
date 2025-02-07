//
//
/*!
  \file
  \ingroup buildblock

  \brief Import of std::shared_ptr, std::dynamic_pointer_cast and
  std::static_pointer_cast into the stir namespace.
*/
/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SHARED_PTR__
#define __stir_SHARED_PTR__

#include "stir/common.h"
#include <memory>

namespace stir
{
using std::shared_ptr;
using std::dynamic_pointer_cast;
using std::static_pointer_cast;
//! work-around for using std::make_shared on old compilers (deprecated)
#define MAKE_SHARED std::make_shared
} // namespace stir

#endif
