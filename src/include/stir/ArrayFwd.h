
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup Array
  \brief forward declaration of stir::Array class for multi-dimensional (numeric) arrays
*/
#include "stir/common.h"

namespace stir
{
template <int num_dimensions, typename elemT>
class Array;

//! type alias for future-proofing for "large" rectangular arrays
template <int num_dimensions, typename elemT>
using ArrayType = Array<num_dimensions, elemT>;
} // namespace stir
