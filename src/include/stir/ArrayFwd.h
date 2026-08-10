
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef ___stir_ArrayFwd_H___
#define ___stir_ArrayFwd_H___
/*!
  \file
  \ingroup Array
  \brief forward declaration of stir::Array class for multi-dimensional (numeric) arrays
*/
#include "stir/VectorWithOffsetFwd.h"
#include "stir/IndexRangeFwd.h"
#include "stir/BasicCoordinateFwd.h"

namespace stir
{
template <int num_dimensions, typename elemT, typename indexT = int>
class Array;

//! type alias for future-proofing for "large" rectangular arrays
template <int num_dimensions, typename elemT, typename indexT = int>
using ArrayType = Array<num_dimensions, elemT, indexT>;

} // namespace stir

#endif
