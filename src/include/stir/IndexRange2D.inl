//
//
/*!
  \file
  \ingroup Array
  \brief  inline implementations for stir::IndexRange2D.

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/Coordinate2D.h"

START_NAMESPACE_STIR

IndexRange2D::IndexRange2D()
    : base_type()
{}

IndexRange2D::IndexRange2D(const IndexRange<2>& range_v)
    : base_type(range_v)
{}

IndexRange2D::IndexRange2D(const int min_1, const int max_1, const int min_2, const int max_2)
    : base_type(Coordinate2D<int>(min_1, min_2), Coordinate2D<int>(max_1, max_2))
{}

IndexRange2D::IndexRange2D(const int length_1, const int length_2)
    : base_type(Coordinate2D<int>(0, 0), Coordinate2D<int>(length_1 - 1, length_2 - 1))
{}

END_NAMESPACE_STIR
