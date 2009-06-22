#ifndef __CartesianCoordinate3D_H__
#define __CartesianCoordinate3D_H__
//
// $Id$
//
/*!
  \file 
  \ingroup Coordinate 
  \brief defines the stir::CartesianCoordinate3D<coordT> class 

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


#include "stir/Coordinate3D.h"

#ifdef STIR_USE_NO_TYPENAME
/* horrible work-around for compilers that do not handle 'typename'
   correctly.
   This work-around relies on the fact that you do not use 'typename'
   in the template parameters (use 'class' instead)
*/
#define typename
#endif

START_NAMESPACE_STIR

/*!
  \ingroup Coordinate
  \brief a templated class for 3-dimensional coordinates.

  It is derived from Coordinate3D<coordT>. The only new methods are
   z(),y(),x(), corresponding resp. to 
   operator[](1), operator[](2), operator[](3)

   \warning The constructor uses the order CartesianCoordinate3D<coordT>(z,y,x)
*/


template <class coordT>
class CartesianCoordinate3D : public Coordinate3D<coordT>
{
protected:
  typedef Coordinate3D<coordT> base_type;
  typedef typename base_type::base_type basebase_type;

public:
  inline CartesianCoordinate3D();
  inline CartesianCoordinate3D(const coordT&, const coordT&, const coordT&);
  inline CartesianCoordinate3D(const basebase_type& c);
  inline CartesianCoordinate3D& operator=(const basebase_type& c);
#ifdef OLDDESIGN
  inline CartesianCoordinate3D(const Point3D& p);
#endif

  inline coordT& z();
  inline coordT z() const;
  inline coordT& y();
  inline coordT y() const;
  inline coordT& x();
  inline coordT x() const;

};

END_NAMESPACE_STIR

#include "stir/CartesianCoordinate3D.inl"

#endif

#ifdef STIR_USE_NO_TYPENAME
#undef typename
#endif
