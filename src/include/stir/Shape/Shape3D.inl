//
// $Id$
//
/*
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
/*!
  \file
  \ingroup Shape

  \brief Inline-implementations of class stir::Shape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

Shape3D::Shape3D()
: origin(0,0,0)
{}

Shape3D::Shape3D(const CartesianCoordinate3D<float>& origin)
: origin(origin)
{}

bool
Shape3D::
operator==(const Shape3D& s) const
{
  return norm(this->origin - s.origin) < .001F;
}


bool
Shape3D::
operator!=(const Shape3D& s) const
{ 
  return !(*this == s);
}
  
void 
Shape3D::scale_around_origin(const CartesianCoordinate3D<float>& scale3D)
{
  CartesianCoordinate3D<float> old_origin = get_origin();
  translate(old_origin * (-1));
  scale(scale3D);
  translate(old_origin);
  assert((norm(get_origin())==0 && norm(old_origin)==0)||
	     norm(get_origin() - old_origin) < norm(get_origin())*10E-5);
}
  
CartesianCoordinate3D<float> Shape3D::get_origin() const
{ return origin; }


END_NAMESPACE_STIR
