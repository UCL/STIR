//
//
#ifndef __Coordinate2D_H__
#define __Coordinate2D_H__
/*!
  \file 
  \ingroup Coordinate 
  \brief defines the stir::Coordinate2D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR


/*!
  \ingroup Coordinate 
  \brief a templated class for 2-dimensional coordinates.

   The only new method is a constructor Coordinate2D<coordT>(c1,c2)

   \warning Indices run from 1 to 2. 

*/

template <typename coordT>
class Coordinate2D : public BasicCoordinate<2, coordT>
{
protected:
  typedef BasicCoordinate<2, coordT> base_type;

public:
  inline Coordinate2D();
  inline Coordinate2D(const coordT&, const coordT&);
  inline Coordinate2D(const BasicCoordinate<2,coordT>& c);
  inline Coordinate2D& operator=(const BasicCoordinate<2, coordT>& c);
};

END_NAMESPACE_STIR

#include "stir/Coordinate2D.inl"

#endif

