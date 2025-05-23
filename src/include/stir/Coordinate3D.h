#ifndef __Coordinate3D_H__
#define __Coordinate3D_H__
//
//
/*!
  \file 
  \ingroup Coordinate 
  \brief defines the stir::Coordinate3D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project



*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/


#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

/*!  \ingroup Coordinate 
   \brief a templated class for 3-dimensional coordinates.

   The only new method is a constructor Coordinate3D<coordT>(c1,c2,c3)

   \warning Indices run from 1 to 3.

*/

template <typename coordT>
class Coordinate3D : public BasicCoordinate<3, coordT>
{
protected:
  typedef BasicCoordinate<3, coordT> base_type;

public:
  inline Coordinate3D();
  inline Coordinate3D(const coordT&, const coordT&, const coordT&);
  inline Coordinate3D(const BasicCoordinate<3, coordT>& c);
  inline Coordinate3D& operator=(const BasicCoordinate<3, coordT>& c);
};

END_NAMESPACE_STIR

#include "stir/Coordinate3D.inl"

#endif

