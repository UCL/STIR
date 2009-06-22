#ifndef __CartesianCoordinate2D_H__
#define __CartesianCoordinate2D_H__
//
// $Id$
//
/*!
  \file 
  \ingroup Coordinate  
  \brief defines the stir::CartesianCoordinate2D<coordT> class 

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


#include "stir/Coordinate2D.h"


START_NAMESPACE_STIR

/*!
  \ingroup Coordinate
   \brief a templated class for 2-dimensional coordinates.

   It is derived from Coordinate2D<coordT>. The only new methods are
   y(),x(), corresponding resp. to 
   operator[](1), operator[](2)

   \warning The constructor uses the order CartesianCoordinate2D<coordT>(y,x)
*/

template <typename coordT>
class CartesianCoordinate2D : public Coordinate2D<coordT>
{
protected:
  typedef Coordinate2D<coordT> base_type;
  typedef typename base_type::base_type basebase_type;

public:
  inline CartesianCoordinate2D();
  inline CartesianCoordinate2D(const coordT&, const coordT&);
  inline CartesianCoordinate2D(const basebase_type& c);
  inline CartesianCoordinate2D& operator=(const basebase_type& c);

  inline coordT& y();
  inline coordT y() const;
  inline coordT& x();
  inline coordT x() const;

};

END_NAMESPACE_STIR

#include "stir/CartesianCoordinate2D.inl"

#endif

