//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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

  \brief Non-inline implementations for class stir::Ellipsoid

  \author Sanida Mustafovic
*/
#include "stir/Shape/Ellipsoid.h"
#include "stir/numerics/MatrixFunction.h"
#include "stir/Succeeded.h"
#include <cmath>

START_NAMESPACE_STIR

const char * const 
Ellipsoid::registered_name = "Ellipsoid";


void 
Ellipsoid::initialise_keymap()
{
  parser.add_start_key("Ellipsoid Parameters");
  parser.add_key("radius-x (in mm)", &radii.x());
  parser.add_key("radius-y (in mm)", &radii.y());
  parser.add_key("radius-z (in mm)", &radii.z());
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}



void
Ellipsoid::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  radii.fill(0);
}


bool
Ellipsoid::
post_processing()
{
  if (Shape3DWithOrientation::post_processing()==true)
    return true;

  if (radii.x() <= 0)
    {
      warning("radii.x() should be positive, but is %g\n", radii.x());
      return true;
    }
  if (radii.y() <= 0)
    {
      warning("radii.y() should be positive, but is %g\n", radii.y());
      return true;
    }
  if (radii.z() <= 0)
    {
      warning("radii.z() should be positive, but is %g\n", radii.z());
      return true;
    }
  return false;
}

Ellipsoid::Ellipsoid()
{
  set_defaults();
}

Ellipsoid::Ellipsoid(const CartesianCoordinate3D<float>& radii_v, 
                     const CartesianCoordinate3D<float>& centre_v,
	             const Array<2,float>& direction_vectors) 
  : 
  radii(radii_v)
{
  assert(radii.x() > 0);
  assert(radii.y() > 0);
  assert(radii.z() > 0);
  this->set_origin(centre_v);
  if (this->set_direction_vectors(direction_vectors) == Succeeded::no)
    error("Ellipsoid constructor called with wrong direction_vectors");
}		     
		     
void
Ellipsoid::
set_radii(const CartesianCoordinate3D<float>& new_radii)
{
  radii = new_radii; 
  assert(radii.x() > 0);
  assert(radii.y() > 0);
  assert(radii.z() > 0);
}

float Ellipsoid::get_geometric_volume() const
 {
   return static_cast<float>((4*radii.x()*radii.y()*radii.z()*_PI)/3) / get_volume_of_unit_cell();
 }

#if 0
// formula is incorrect except when it's a sphere
// also, scaling of axes does not simply scale area
float 
Ellipsoid:: 
get_geometric_area()const
{
  return static_cast<float>(4*std::pow(radii.x()*radii.y()*radii.z() / get_volume_of_unit_cell(),2.F/3)*_PI);
}
#endif

bool Ellipsoid::is_inside_shape(const CartesianCoordinate3D<float>& coord) const

{
  const CartesianCoordinate3D<float> r = 
    this->transform_to_shape_coords(coord);
  
   if (norm_squared(r / this->radii)<=1)
      return true;    
   else 
      return false;
}

 
Shape3D* Ellipsoid:: clone() const
{
  return static_cast<Shape3D *>(new Ellipsoid(*this));
}

bool
Ellipsoid:: 
operator==(const Ellipsoid& cylinder) const
{
  const float tolerance = 
    std::min(radii.z(), std::min(radii.x(), radii.y()))/1000;
  return
    std::fabs(this->radii.x() - cylinder.radii.x()) < tolerance
    && std::fabs(this->radii.y() - cylinder.radii.y()) < tolerance
    && std::fabs(this->radii.z() - cylinder.radii.z()) < tolerance
    && Shape3DWithOrientation::operator==(cylinder);
;
}

bool
Ellipsoid:: 
operator==(const Shape3D& shape) const
{
  Ellipsoid const * cylinder_ptr =
    dynamic_cast<Ellipsoid const *>(&shape);
  return
    cylinder_ptr != 0 && (*this == *cylinder_ptr);
}


END_NAMESPACE_STIR
