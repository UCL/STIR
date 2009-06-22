//
// $Id$
//
#ifndef __Coordinate4D_H__
#define __Coordinate4D_H__
/*!
  \file 
  \ingroup Coordinate 
  \brief defines the stir::Coordinate4D<coordT> class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/



#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR
/*!
  \ingroup Coordinate 
  \brief a templated class for 4-dimensional coordinates.

   The only new method is a constructor Coordinate4D<coordT>(c1,c2,c3,c4)

   \warning Indices run from 1 to 4. 

*/
template <typename coordT>
class Coordinate4D : public BasicCoordinate<4, coordT>
{
protected:
  typedef BasicCoordinate<4, coordT> base_type;

public:
  inline Coordinate4D();
  inline Coordinate4D(const coordT&, const coordT&, const coordT&, const coordT&);
  inline Coordinate4D(const base_type& c);
  inline Coordinate4D& operator=(const base_type& c);
};

END_NAMESPACE_STIR

#include "stir/Coordinate4D.inl"

#endif

