//
//
/*!
  \file
  \ingroup Array

  \brief  inline implementations for stir::IndexRange4D.

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

#include "stir/Coordinate4D.h"

START_NAMESPACE_STIR

IndexRange4D::IndexRange4D()
    : base_type()
{}

IndexRange4D::IndexRange4D(const IndexRange<4>& range_v)
    : base_type(range_v)
{}

IndexRange4D::IndexRange4D(const int min_1,
                           const int max_1,
                           const int min_2,
                           const int max_2,
                           const int min_3,
                           const int max_3,
                           const int min_4,
                           const int max_4)
    : base_type(Coordinate4D<int>(min_1, min_2, min_3, min_4), Coordinate4D<int>(max_1, max_2, max_3, max_4))
{}

IndexRange4D::IndexRange4D(const int length_1, const int length_2, const int length_3, const int length_4)
    : base_type(Coordinate4D<int>(0, 0, 0, 0), Coordinate4D<int>(length_1 - 1, length_2 - 1, length_3 - 1, length_4 - 1))
{}
END_NAMESPACE_STIR
