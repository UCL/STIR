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

  \brief Non-inline implementations for class stir::EllipsoidalCylinder

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/numerics/MatrixFunction.h"
#include "stir/Succeeded.h"
#include <algorithm>
#include <cmath>

START_NAMESPACE_STIR

const char * const 
EllipsoidalCylinder::registered_name = "Ellipsoidal Cylinder";

void 
EllipsoidalCylinder::initialise_keymap()
{
  parser.add_start_key("Ellipsoidal Cylinder Parameters");
  parser.add_key("radius-x (in mm)", &radius_x);
  parser.add_key("radius-y (in mm)", &radius_y);
  parser.add_key("length-z (in mm)", &length);
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}



void
EllipsoidalCylinder::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  radius_x=0;
  radius_y=0;
  length=0;
}

bool
EllipsoidalCylinder::
post_processing()
{
  if (Shape3DWithOrientation::post_processing()==true)
    return true;

  if (radius_x <= 0)
    {
      warning("radius_x should be positive, but is %g\n", radius_x);
      return true;
    }
  if (radius_y <= 0)
    {
      warning("radius_y should be positive, but is %g\n", radius_y);
      return true;
    }
  return false;
}



EllipsoidalCylinder::EllipsoidalCylinder()
{
  set_defaults(); 
}

EllipsoidalCylinder::EllipsoidalCylinder(const float length_v, 
                         const float radius_xv,
	                 const float radius_yv,
	                 const CartesianCoordinate3D<float>& centre_v,
	                 const Array<2,float>& direction_vectors) 
                    :
                     length(length_v),
		     radius_x(radius_xv),
		     radius_y(radius_yv)
		    
{
  assert(length>0);
  assert(radius_x>0);
  assert(radius_y>0);
  this->set_origin(centre_v);
  if (this->set_direction_vectors(direction_vectors) == Succeeded::no)
    error("Ellipsoid constructor called with wrong direction_vectors");
}  

void
EllipsoidalCylinder::
set_length(const float new_length)
{
  assert(new_length>0);
  length = new_length;
}

void
EllipsoidalCylinder::
set_radius_x(const float new_radius_x)
{
  assert(new_radius_x>0);
  radius_x = new_radius_x;
}

void
EllipsoidalCylinder::
set_radius_y(const float new_radius_y)
{
  assert(new_radius_y>0);
  radius_y = new_radius_y;
}



bool EllipsoidalCylinder::is_inside_shape(const CartesianCoordinate3D<float>& coord) const

{
  const CartesianCoordinate3D<float> r = 
    this->transform_to_shape_coords(coord);
  
  const float distance_along_axis= r.z();
  
  if (fabs(distance_along_axis)<length/2)
  { 
    if (square(r.x()/radius_x) + square(r.y()/radius_y) <=1)
      return true;
    else 
      return false;
  }
  else return false;
}

float 
EllipsoidalCylinder:: 
get_geometric_volume()const
 {
   return static_cast<float>(radius_x*radius_y*_PI*length) / get_volume_of_unit_cell();
 }

#if 0
// formula is incorrect (does not include end planes, and does not handle ellips)
// also, scaling of axes does not simply scale area
float 
EllipsoidalCylinder:: 
get_geometric_area()const
{
  return static_cast<float>(2*sqrt(radius_x*radius_y)*_PI*length) / get_volume_of_unit_cell();
}
#endif

Shape3D* 
EllipsoidalCylinder:: 
clone() const
{
  return static_cast<Shape3D *>(new EllipsoidalCylinder(*this));
}

bool
EllipsoidalCylinder:: 
operator==(const EllipsoidalCylinder& cylinder) const
{
  const float tolerance = 
    std::min(length, std::min(radius_x, radius_y))/1000;
  return
    std::fabs(this->length - cylinder.length) < tolerance
    && std::fabs(this->radius_x - cylinder.radius_x) < tolerance
    && std::fabs(this->radius_y - cylinder.radius_y) < tolerance
    && Shape3DWithOrientation::operator==(cylinder);

;
}

bool
EllipsoidalCylinder:: 
operator==(const Shape3D& shape) const
{
  EllipsoidalCylinder const * cylinder_ptr =
    dynamic_cast<EllipsoidalCylinder const *>(&shape);
  return
    cylinder_ptr != 0 && (*this == *cylinder_ptr);
}

END_NAMESPACE_STIR
