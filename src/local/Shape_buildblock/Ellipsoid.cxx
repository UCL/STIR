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
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/Shape/Ellipsoid.h"

START_NAMESPACE_STIR

Ellipsoid::Ellipsoid()
{
  set_defaults();
}

Ellipsoid::Ellipsoid(const float radius_av, 
                      const float radius_bv,
	              const float radius_cv,
	              const CartesianCoordinate3D<float>& centre_v,
		      const float alpha_v,
		      const float beta_v,                     
		      const float gamma_v) 
                     ://Shape3DWithOrientation(centre_v,alpha_v,beta_v,gamma_v),
                     radius_a(radius_av),
		     radius_b(radius_bv),
		     radius_c(radius_cv)		    
                          
{
  origin = centre_v;
  set_directions_from_Euler_angles(alpha_v, beta_v, gamma_v);
}

 // dir_a, dir_b, dir_c must be orthogonal to each other

Ellipsoid::Ellipsoid(const float radius_av, 
                     const float radius_bv,
	             const float radius_cv,
	             const CartesianCoordinate3D<float>& centre_v,
	             const CartesianCoordinate3D<float>& dir_av,
		     const CartesianCoordinate3D<float>& dir_bv,
		     const CartesianCoordinate3D<float>& dir_cv) 
                    : //Shape3DWithOrientation(centre_v,dir_av,dir_bv,dir_cv),
		     radius_a(radius_av),
		     radius_b(radius_bv),
		     radius_c(radius_cv)
{
  origin = centre_v;
  dir_x = dir_av;
  dir_y = dir_bv;
  dir_z = dir_cv;
}		     
		     

float Ellipsoid::get_geometric_volume() const
 {
   return ((4*radius_a*radius_b*radius_c*_PI)/3);
 }


bool Ellipsoid::is_inside_shape(const CartesianCoordinate3D<float>& index) const

{
  assert(radius_a > 0);
  assert(radius_b > 0);
  assert(radius_c > 0);

  CartesianCoordinate3D<float> r = index - origin;
  
   if ((square(inner_product(r,dir_x))/square(radius_a)+ 
        square(inner_product(r,dir_y))/square(radius_b)+
        square(inner_product(r,dir_z))/square(radius_c))<=1)
      return true;    
   else 
      return false;
}

 
Shape3D* Ellipsoid:: clone() const
{
  return static_cast<Shape3D *>(new Ellipsoid(*this));
}


void 
Ellipsoid::initialise_keymap()
{
  parser.add_start_key("Ellipsoid Parameters");
  parser.add_key("radius-x (in mm)", &radius_a);
  parser.add_key("radius-y (in mm)", &radius_b);
  parser.add_key("radius-z (in mm)", &radius_c);
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}



void
Ellipsoid::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  radius_a=0;
  radius_b=0;
  radius_c=0;
}


const char * const 
Ellipsoid::registered_name = "Ellipsoid";

END_NAMESPACE_STIR
