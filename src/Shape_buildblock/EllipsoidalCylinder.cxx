//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class EllipsoidalCylinder

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/Shape/EllipsoidalCylinder.h"

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
	                 const CartesianCoordinate3D<float>& dir_xv,
			 const CartesianCoordinate3D<float>& dir_yv,
			 const CartesianCoordinate3D<float>& dir_zv) 
                    ://Shape3DWithOrientation(centre_v,dir_xv,dir_yv,dir_zv), 
                     length(length_v),
		     radius_x(radius_xv),
		     radius_y(radius_yv)
		    
{
  origin = centre_v;
  dir_x = dir_xv;
  dir_y = dir_yv;
  dir_z = dir_zv;
}  



EllipsoidalCylinder::EllipsoidalCylinder(const float length_v, 
                     const float radius_xv,
	             const float radius_yv,
	             const CartesianCoordinate3D<float>& centre_v,
		     const float alpha_v,
	             const float beta_v,
                     const float gamma_v) 
		    ://Shape3DWithOrientation(centre_v,alpha_v,beta_v,gamma_v),
                     length(length_v),
		     radius_x(radius_xv),
		     radius_y(radius_yv)		    
{
  origin = centre_v;
  set_directions_from_Euler_angles(alpha_v, beta_v, gamma_v);
}


bool EllipsoidalCylinder::is_inside_shape(const CartesianCoordinate3D<float>& index) const

{

  const CartesianCoordinate3D<float> r = index - origin;
  
  const float distance_along_axis=
      inner_product(r,dir_z);
  
  if (fabs(distance_along_axis)<length/2)
  { 
    if ((square(inner_product(r,dir_x))/square(radius_x) + 
         square(inner_product(r,dir_y))/square(radius_y))<=1)
      return true;
    else 
      return false;
  }
  else return false;
}


void 
EllipsoidalCylinder::scale(const CartesianCoordinate3D<float>& scale3D)
{
  if (norm(dir_z - CartesianCoordinate3D<float>(1,0,0)) > 1E-5F ||
      norm(dir_y - CartesianCoordinate3D<float>(0,1,0)) > 1E-5F ||	
      norm(dir_x - CartesianCoordinate3D<float>(0,0,1)) > 1E-5F)
    error("EllipsoidalCylinder::scale cannot handle rotated case yet.\n");
  // TODO it's probably better to scale dir_x et al, but then other things might brake  (such as geometric_volume)

  origin *= scale3D;
  length *= scale3D.z();
  radius_y *= scale3D.y();
  radius_x *= scale3D.x();
}

float 
EllipsoidalCylinder:: 
get_geometric_volume()const
 {
   return static_cast<float>(radius_x*radius_y*_PI*length);
 }


float 
EllipsoidalCylinder:: 
get_geometric_area()const
{
  return static_cast<float>(2*sqrt(radius_x*radius_y)*_PI*length);
}

Shape3D* 
EllipsoidalCylinder:: 
clone() const
{
  return static_cast<Shape3D *>(new EllipsoidalCylinder(*this));
}

END_NAMESPACE_STIR
