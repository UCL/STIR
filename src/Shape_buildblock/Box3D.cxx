//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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

  \brief Non-inline implementations for class stir::Box3D

  \author C. Ross Schmidtlein
  \author Kris Thielemans (brought up-to-date to STIR 2.0)
*/
#include "stir/Shape/Box3D.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const 
Box3D::registered_name = "Box3D";

void 
Box3D::initialise_keymap()
{
  parser.add_start_key("box parameters");
  parser.add_key("length-x (in mm)", &length_x);
  parser.add_key("length-y (in mm)", &length_y);
  parser.add_key("length-z (in mm)", &length_z);
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}

void
Box3D::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  length_x=0;
  length_y=0;
  length_z=0;
}

bool
Box3D::
post_processing()
{
  if (Shape3DWithOrientation::post_processing()==true)
    return true;

  if (length_x <= 0)
    {
      warning("length_x should be positive, but is %g\n", length_x);
      return true;
    }
  if (length_y <= 0)
    {
      warning("length_y should be positive, but is %g\n", length_y);
      return true;
    }
  if (length_z <= 0)
    {
      warning("length_z should be positive, but is %g\n", length_z);
      return true;
    }
  return false;
}

Box3D::Box3D()
{
  set_defaults(); 
}

Box3D::Box3D(const float length_xv, 
	     const float length_yv,
	     const float length_zv,
	     const CartesianCoordinate3D<float>& centre_v,
	     const Array<2,float>& direction_vectors) 
  :
  length_x(length_xv),
  length_y(length_yv),
  length_z(length_zv)
{
  this->set_origin(centre_v);
  if (this->set_direction_vectors(direction_vectors) == Succeeded::no)
    error("Box3D constructor called with wrong direction_vectors");
}  
#if 0

Box3D::Box3D(const float length_xv,
	     const float length_yv,
	     const float length_zv,
	     const CartesianCoordinate3D<float>& centre_v,
	     const float alpha_v,
	     const float beta_v,
	     const float gamma_v) 
  :
  length_x(length_xv),
  length_y(length_yv),
  length_z(length_zv)
{
  this->set_origin(centre_v);
  this->set_directions_from_Euler_angles(alpha_v, beta_v, gamma_v);
}

#endif

bool Box3D::is_inside_shape(const CartesianCoordinate3D<float>& coord) const
{
  const CartesianCoordinate3D<float> r = 
    this->transform_to_shape_coords(coord);
  
  const float distance_along_x_axis= r.x();
  const float distance_along_y_axis= r.y();
  const float distance_along_z_axis= r.z();
  
  return
    fabs(distance_along_x_axis)<length_x/2
    && fabs(distance_along_y_axis)<length_y/2
    && fabs(distance_along_z_axis)<length_z/2;
}

float 
Box3D:: 
get_geometric_volume()const
{
   return static_cast<float>(length_x*length_y*length_z) / this->get_volume_of_unit_cell();
}


#if 0
// doesn't take scaling into account
float 
Box3D:: 
get_geometric_area()const
{
  return static_cast<float>(2*(length_x*length_y+length_x
			       *length_z+length_y*length_z));
}
#endif

Shape3D* 
Box3D:: 
clone() const
{
  return static_cast<Shape3D *>(new Box3D(*this));
}

bool
Box3D:: 
operator==(const Box3D& box) const
{
  const float tolerance = 
    std::min(length_z, std::min(length_x, length_y))/1000;
  return
    std::fabs(this->length_x - box.length_x) < tolerance
    && std::fabs(this->length_y - box.length_y) < tolerance
    && std::fabs(this->length_z - box.length_z) < tolerance
    && Shape3DWithOrientation::operator==(box);
}

bool
Box3D:: 
operator==(const Shape3D& shape) const
{
  Box3D const * box_ptr =
    dynamic_cast<Box3D const *>(&shape);
  return
    box_ptr != 0 && (*this == *box_ptr);
}

END_NAMESPACE_STIR
