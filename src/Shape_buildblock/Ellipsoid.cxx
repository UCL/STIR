//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class Ellipsoid

  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/Shape/Ellipsoid.h"
#include <math.h>

START_NAMESPACE_STIR

const char * const 
Ellipsoid::registered_name = "Ellipsoid";


void 
Ellipsoid::initialise_keymap()
{
  parser.add_start_key("Ellipsoid Parameters");
  parser.add_key("radius-x (in mm)", &radius_x);
  parser.add_key("radius-y (in mm)", &radius_y);
  parser.add_key("radius-z (in mm)", &radius_z);
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}



void
Ellipsoid::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  radius_x=0;
  radius_y=0;
  radius_z=0;
}


bool
Ellipsoid::
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
  if (radius_z <= 0)
    {
      warning("radius_z should be positive, but is %g\n", radius_z);
      return true;
    }
  return false;
}

Ellipsoid::Ellipsoid()
{
  set_defaults();
}

Ellipsoid::Ellipsoid(const float radius_xv, 
                      const float radius_yv,
	              const float radius_zv,
	              const CartesianCoordinate3D<float>& centre_v,
		      const float alpha_v,
		      const float beta_v,                     
		      const float gamma_v) 
                     ://Shape3DWithOrientation(centre_v,alpha_v,beta_v,gamma_v),
                     radius_x(radius_xv),
		     radius_y(radius_yv),
		     radius_z(radius_zv)		    
                          
{
  origin = centre_v;
  set_directions_from_Euler_angles(alpha_v, beta_v, gamma_v);
}

 // dir_a, dir_b, dir_c must be orthogonal to each other

Ellipsoid::Ellipsoid(const float radius_xv, 
                     const float radius_yv,
	             const float radius_zv,
	             const CartesianCoordinate3D<float>& centre_v,
	             const CartesianCoordinate3D<float>& dir_av,
		     const CartesianCoordinate3D<float>& dir_bv,
		     const CartesianCoordinate3D<float>& dir_cv) 
                    : //Shape3DWithOrientation(centre_v,dir_av,dir_bv,dir_cv),
		     radius_x(radius_xv),
		     radius_y(radius_yv),
		     radius_z(radius_zv)
{
  origin = centre_v;
  dir_x = dir_av;
  dir_y = dir_bv;
  dir_z = dir_cv;
}		     
		     

float Ellipsoid::get_geometric_volume() const
 {
   return static_cast<float>((4*radius_x*radius_y*radius_z*_PI)/3);
 }


float 
Ellipsoid:: 
get_geometric_area()const
{
  return static_cast<float>(4*pow(radius_x*radius_y*radius_z,2.F/3)*_PI);
}

bool Ellipsoid::is_inside_shape(const CartesianCoordinate3D<float>& index) const

{
  assert(radius_x > 0);
  assert(radius_y > 0);
  assert(radius_z > 0);

  CartesianCoordinate3D<float> r = index - origin;
  
   if ((square(inner_product(r,dir_x))/square(radius_x)+ 
        square(inner_product(r,dir_y))/square(radius_y)+
        square(inner_product(r,dir_z))/square(radius_z))<=1)
      return true;    
   else 
      return false;
}

 
Shape3D* Ellipsoid:: clone() const
{
  return static_cast<Shape3D *>(new Ellipsoid(*this));
}



END_NAMESPACE_STIR
