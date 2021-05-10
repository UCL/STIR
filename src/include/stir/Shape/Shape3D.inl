//
//
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Shape

  \brief Inline-implementations of class stir::Shape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
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
