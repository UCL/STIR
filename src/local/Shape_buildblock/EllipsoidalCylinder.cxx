//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class EllipsoidalCylinder

  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/Shape/EllipsoidalCylinder.h"

START_NAMESPACE_STIR

EllipsoidalCylinder::EllipsoidalCylinder()
{
  set_defaults(); 
}

EllipsoidalCylinder::EllipsoidalCylinder(const float length_v, 
                         const float radius_av,
	                 const float radius_bv,
	                 const CartesianCoordinate3D<float>& centre_v,
	                 const CartesianCoordinate3D<float>& dir_xv,
			 const CartesianCoordinate3D<float>& dir_yv,
			 const CartesianCoordinate3D<float>& dir_zv) 
                    ://Shape3DWithOrientation(centre_v,dir_xv,dir_yv,dir_zv), 
                     length(length_v),
		     radius_a(radius_av),
		     radius_b(radius_bv)
		    
{
  origin = centre_v;
  dir_x = dir_xv;
  dir_y = dir_yv;
  dir_z = dir_zv;
}  



EllipsoidalCylinder::EllipsoidalCylinder(const float length_v, 
                     const float radius_av,
	             const float radius_bv,
	             const CartesianCoordinate3D<float>& centre_v,
		     const float alpha_v,
	             const float beta_v,
                     const float gamma_v) 
		    ://Shape3DWithOrientation(centre_v,alpha_v,beta_v,gamma_v),
                     length(length_v),
		     radius_a(radius_av),
		     radius_b(radius_bv)		    
{
  origin = centre_v;
  set_directions_from_Euler_angles(alpha_v, beta_v, gamma_v);
}


//This method determines if the current value is the part 
// of the cylinder, returns 1 if it is true

bool EllipsoidalCylinder::is_inside_shape(const CartesianCoordinate3D<float>& index) const

{

  const CartesianCoordinate3D<float> r = index - origin;
  
  const float distance_along_axis=
      inner_product(r,dir_z);
  
  if (fabs(distance_along_axis)<length/2)
  { 
    if ((square(inner_product(r,dir_x))/square(radius_a) + 
         square(inner_product(r,dir_y))/square(radius_b))<=1)
      return true;
    else 
      return false;
  }
  else return false;
}

void 
EllipsoidalCylinder::initialise_keymap()
{
  parser.add_start_key("Ellipsoidal Cylinder Parameters");
  parser.add_key("radius-x (in mm)", &radius_a);
  parser.add_key("radius-y (in mm)", &radius_b);
  parser.add_key("length-z (in mm)", &length);
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}



void
EllipsoidalCylinder::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  radius_a=0;
  radius_b=0;
  length=0;
}


const char * const 
EllipsoidalCylinder::registered_name = "Ellipsoidal Cylinder";


END_NAMESPACE_STIR
