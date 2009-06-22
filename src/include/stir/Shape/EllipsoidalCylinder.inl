//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Inline-implementations of class stir::EllipsoidalCylinder

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
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
START_NAMESPACE_STIR

float EllipsoidalCylinder:: get_geometric_volume()const
 {
   return (radius_a*radius_b*_PI*length);
 }

Shape3D* EllipsoidalCylinder:: clone() const
{
  return static_cast<Shape3D *>(new EllipsoidalCylinder(*this));
}
void 
EllipsoidalCylinder::scale(const CartesianCoordinate3D<float>& scale3D)
{
  origin *= scale3D;
  length *= scale3D.z();
  radius_b *= scale3D.y();
  radius_a *= scale3D.x();
}

END_NAMESPACE_STIR
