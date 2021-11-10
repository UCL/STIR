#ifndef __stir_cross_product_H__
#define __stir_cross_product_H__
//
//
/*
    Copyright (C) 2003- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup buildblock 
  \brief defines the cross-product of 2 CartesianCoordinate3D numbers

  \author Kris Thielemans 

*/

#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief the cross-product for 3-dimensional coordinates.

  \warning This implements <strong>minus</strong> the 'usual' definition 
  of the cross-product. This is done because STIR uses a left-handed 
  coordinate system. The definition of \a cross_product is such that 
  \f$ {a, b, a\times b}\f$ forms a <i>left</i>-handed
  coordinate system.
*/


template <class coordT>
CartesianCoordinate3D<coordT>
cross_product(const CartesianCoordinate3D<coordT>& a,
	      const CartesianCoordinate3D<coordT>& b)
{
  return
    CartesianCoordinate3D<coordT>(a.y()*b.x() - a.x()*b.y(),
				  -a.z()*b.x() + a.x()*b.z(),
				  a.z()*b.y() - a.y()*b.z());
};

END_NAMESPACE_STIR

#endif
