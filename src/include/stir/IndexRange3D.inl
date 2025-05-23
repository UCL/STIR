//
//
/*! 
  \file
  \ingroup Array
  \brief  inline implementations for stir::IndexRange3D.

  \author Sanida Mustafovic
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

#include "stir/Coordinate3D.h"

START_NAMESPACE_STIR

IndexRange3D::IndexRange3D()
: base_type()
{}


IndexRange3D::IndexRange3D(const IndexRange<3>& range_v)
: base_type(range_v)
{}

IndexRange3D::IndexRange3D(const int min_1, const int max_1,
                      const int min_2, const int max_2,
                      const int min_3, const int max_3)
			  :base_type(Coordinate3D<int>(min_1,min_2,min_3),
			             Coordinate3D<int>(max_1,max_2,max_3))
{}
 
IndexRange3D::IndexRange3D(const int length_1, const int length_2, const int length_3)
: base_type(Coordinate3D<int>(0,0,0),
	    Coordinate3D<int>(length_1-1,length_2-1,length_3-1))
{}
END_NAMESPACE_STIR 
