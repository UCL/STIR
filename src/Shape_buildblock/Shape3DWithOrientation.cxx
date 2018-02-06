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

  \brief Non-inline implementations for class stir::Shape3DWithOrientation

  \author Sanida Mustafovic
  \author Kris Thielemans
*/
#include "stir/Shape/Shape3DWithOrientation.h"
#include "stir/numerics/determinant.h"
#include "stir/numerics/norm.h"
#include "stir/Succeeded.h"
#include <cmath>

START_NAMESPACE_STIR


#if 0
void 
Shape3DWithOrientation::
set_directions_from_Euler_angles(
                                 const float alpha,
                                 const float beta,
                                 const float gamma)
{
/*
  dir_x( -cos(gamma)*sin(beta),
         cos(beta)*cos(gamma)*sin(alpha)+cos(alpha)*sin(gamma),
         cos(alpha)*cos(beta)*cos(gamma)-sin(alpha)*sin(gamma)),
  dir_y( sin(beta)*sin(gamma),
         cos(alpha)*cos(gamma)-cos(beta)*sin(alpha)*sin(gamma),	 
	 -cos(gamma)*sin(alpha)-cos(alpha)*cos(beta)*sin(gamma)),
  dir_z( sin(beta)*sin(gamma),
         cos(alpha)*cos(gamma)-cos(beta)*sin(alpha)*sin(gamma),
	-cos(gamma)*sin(alpha)-cos(alpha)*cos(beta)*sin(gamma))
   */
   
  dir_x = CartesianCoordinate3D<float>(
	sin(beta)*sin(gamma),
	cos(gamma)*sin(alpha) + cos(alpha)*cos(beta)*sin(gamma),
	cos(alpha)*cos(gamma) - cos(beta)*sin(alpha)*sin(gamma));
  dir_y = CartesianCoordinate3D<float>(
	cos(gamma)*sin(beta),
	cos(alpha)*cos(beta)*cos(gamma) - sin(alpha)*sin(gamma),
	-(cos(beta)*cos(gamma)*sin(alpha)) - cos(alpha)*sin(gamma));
  dir_z = CartesianCoordinate3D<float>(
	cos(beta),
	-(cos(alpha)*sin(beta)),
	sin(alpha)*sin(beta));
}
#endif



Shape3DWithOrientation::Shape3DWithOrientation()
{}


Shape3DWithOrientation::Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
                                               const Array<2,float>& direction_vectors)
: Shape3D(origin)
{
  if (this->set_direction_vectors(direction_vectors) == Succeeded::no)
    error("Shaped3DWithOrientation constructor called with wrong direction_vectors");
}

Succeeded
Shape3DWithOrientation::
set_direction_vectors(const Array<2,float>& directions)
{
  this->_directions = directions;
  if (this->_directions.size() != 3)
    return Succeeded::no;

  // set index offset to 1, such that matrix_multiply can be used with BasicCoordinate
  this->_directions.set_min_index(1);
  for (int i=1; i<=this->_directions.get_max_index(); ++i)
    {
      this->_directions[i].set_min_index(1);
        if (this->_directions[i].size() != 3)
	  return Succeeded::no;
    }
  return Succeeded::yes;
}

bool
Shape3DWithOrientation::
operator==(const Shape3DWithOrientation& s) const
{
  const float tolerance = .001F;
  return 
    norm(this->get_origin() - s.get_origin()) < tolerance
    && norm(this->_directions[1] - s._directions[1]) < tolerance
    && norm(this->_directions[2] - s._directions[2]) < tolerance
    && norm(this->_directions[3] - s._directions[3]) < tolerance
    && base_type::operator==(s);
}

float 
Shape3DWithOrientation::
get_volume_of_unit_cell() const
{
  return std::fabs(determinant(this->get_direction_vectors()));
}

CartesianCoordinate3D<float>
Shape3DWithOrientation::
transform_to_shape_coords(const CartesianCoordinate3D<float>& coord) const
{
  return 
    matrix_multiply(this->get_direction_vectors(), coord - this->get_origin());
}

void Shape3DWithOrientation::scale(const CartesianCoordinate3D<float>& scale3D)
{
  this->_directions[1] /= scale3D[1];
  this->_directions[2] /= scale3D[2];
  this->_directions[3] /= scale3D[3];

  this->set_origin(this->get_origin() * scale3D);
}

#if 0
float Shape3DWithOrientation::get_angle_alpha() const
{
  return atan2(dir_z.y(),dir_z.x());
}

float Shape3DWithOrientation::get_angle_beta()const
{
   return atan2(sqrt(square(dir_z.x())+ square(dir_z.y())),dir_z.z());
}

float Shape3DWithOrientation::get_angle_gamma()const			   
{
  return atan2(-dir_y.z(),_directions.x().z());
}
#endif
  		   
void 
Shape3DWithOrientation::
set_defaults()
{
  Shape3D::set_defaults();
  this->set_direction_vectors(diagonal_matrix(3,1.F));

#if 0
  // set alpha,beta,gamma to non-sensical values for parsing
  // this is necessary because we need to detect if they are used or not
  // see post_processing()
  alpha_in_degrees = beta_in_degrees = gamma_in_degrees = 10000000.F;
#endif
}

void 
Shape3DWithOrientation::
initialise_keymap()
{
  Shape3D::initialise_keymap();
#if 0
  parser.add_key("Euler angle alpha (in degrees)", &alpha_in_degrees);
  parser.add_key("Euler angle beta (in degrees)", &beta_in_degrees);
  parser.add_key("Euler angle gamma (in degrees)", &gamma_in_degrees);
#endif
  parser.add_key("direction vectors (in mm)", &_directions);
}

bool
Shape3DWithOrientation::
post_processing()
{
#if 0
  if (alpha_in_degrees != 10000000.F 
      || beta_in_degrees != 10000000.F 
      || gamma_in_degrees != 10000000.F)
    {
      // one of the Euler angles was set. Now check if all were set
      if (!(alpha_in_degrees != 10000000.F 
	    && beta_in_degrees != 10000000.F 
	    && gamma_in_degrees != 10000000.F))
	{
	  warning("Shape3DWithOrientation: one of the Euler angles was set, but not all");
	  return true;
	}
      set_directions_from_Euler_angles(                                 
				       static_cast<float>(alpha_in_degrees * _PI/180.),
				       static_cast<float>(beta_in_degrees * _PI/180.),
				       static_cast<float>(gamma_in_degrees * _PI/180.));
    }
  else
    {
      // assume that directions have been set
    }
#endif
  // make sure that indices etc are ok
  if (this->set_direction_vectors(_directions) == Succeeded::no)
    {
      warning("Direction vectors should be a 3x3 matrix");
      return true;
    }

  return Shape3D::post_processing();
}

void
Shape3DWithOrientation::
set_key_values()
{
  base_type::set_key_values();
#if 0
  alpha_in_degrees = static_cast<float>(get_angle_alpha()*180./_PI);
  beta_in_degrees = static_cast<float>(get_angle_beta()*180./_PI);
  gamma_in_degrees = static_cast<float>(get_angle_gamma()*180./_PI);
#endif
}
END_NAMESPACE_STIR
