//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class Shape3DWithOrientation

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR


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



Shape3DWithOrientation::Shape3DWithOrientation()
   :Shape3D(), 
    dir_x(0,0,1),
    dir_y(0,1,0),
    dir_z(1,0,0)
{}

Shape3DWithOrientation::Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
                                               const float alpha,
					       const float beta,
					       const float gamma)
: 
  Shape3D(origin)
{
  set_directions_from_Euler_angles(alpha, beta, gamma);
}

Shape3DWithOrientation::Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
                                               const CartesianCoordinate3D<float>& dir_xv,
					       const CartesianCoordinate3D<float>& dir_yv,
					       const CartesianCoordinate3D<float>& dir_zv)
: Shape3D(origin),
  dir_x(dir_xv),
  dir_y(dir_yv),
  dir_z(dir_zv)
{}


void Shape3DWithOrientation::translate(const CartesianCoordinate3D<float>& direction)
{ origin += direction; }


/*
void Shape3DWithOrientation::scale(const CartesianCoordinate3D<float>& scale3D)
{
  dir_x /= scale3D;
  dir_y /= scale3D;
  dir_z /= scale3D;

  origin *= scale3D;
}
*/

CartesianCoordinate3D<float> Shape3DWithOrientation::get_dir_x() const
{
  return dir_x;
}

CartesianCoordinate3D<float> Shape3DWithOrientation::get_dir_y() const
{
  return dir_y;
}

CartesianCoordinate3D<float> Shape3DWithOrientation::get_dir_z() const
{
  return dir_z;
}

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
  return atan2(-dir_y.z(),dir_x.z());
}
  		   
void 
Shape3DWithOrientation::
set_defaults()
{
  Shape3D::set_defaults();
  dir_x = CartesianCoordinate3D<float>(0,0,1);
  dir_y = CartesianCoordinate3D<float>(0,1,0);
  dir_z = CartesianCoordinate3D<float>(1,0,0);

}

void 
Shape3DWithOrientation::
initialise_keymap()
{
  Shape3D::initialise_keymap();
  parser.add_key("Euler angle alpha (in degrees)", &alpha_in_degrees);
  parser.add_key("Euler angle beta (in degrees)", &beta_in_degrees);
  parser.add_key("Euler angle gamma (in degrees)", &gamma_in_degrees);
  parser.add_key("direction-x (in mm)", &dir_x_vector);
  parser.add_key("direction-y (in mm)", &dir_y_vector);
  parser.add_key("direction-z (in mm)", &dir_z_vector);
}

bool
Shape3DWithOrientation::
post_processing()
{
  if (dir_x_vector.size()==0 && dir_y_vector.size()==0 && dir_z_vector.size()==0)
    set_directions_from_Euler_angles(                                 
				     static_cast<float>(alpha_in_degrees * _PI/180.),
				     static_cast<float>(beta_in_degrees * _PI/180.),
				     static_cast<float>(gamma_in_degrees * _PI/180.));
  else if (dir_x_vector.size()==3 && dir_y_vector.size()==3 && dir_z_vector.size()==3)
    {
      
    }
  else
    {
      warning("Either specify Euler angles or the directions (which each be a list of 3 numbers.\n");
      return false;
      dir_x = CartesianCoordinate3D<float>(static_cast<float>(dir_x_vector[0]),
					   static_cast<float>(dir_x_vector[1]),
					   static_cast<float>(dir_x_vector[2]));
      dir_y = CartesianCoordinate3D<float>(static_cast<float>(dir_y_vector[0]),
					   static_cast<float>(dir_y_vector[1]),
					   static_cast<float>(dir_y_vector[2]));
      dir_z = CartesianCoordinate3D<float>(static_cast<float>(dir_z_vector[0]),
					   static_cast<float>(dir_z_vector[1]),
					   static_cast<float>(dir_z_vector[2]));
    }
  return Shape3D::post_processing();
}

void
Shape3DWithOrientation::
set_key_values()
{
  alpha_in_degrees = static_cast<float>(get_angle_alpha()*180./_PI);
  beta_in_degrees = static_cast<float>(get_angle_beta()*180./_PI);
  gamma_in_degrees = static_cast<float>(get_angle_gamma()*180./_PI);
}
END_NAMESPACE_STIR
