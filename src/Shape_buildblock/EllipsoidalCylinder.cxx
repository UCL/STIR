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
  \author C. Ross Schmidtlein (Added new parsing commands and functions so that a 
  sector of a cylinder can be defined and updated the 
  associated tests and volume and area calculations.)_

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
  parser.add_key("initial angle (in deg)", &theta_1);
  parser.add_key("final angle   (in deg)", &theta_2);
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
  theta_1=0.0;
  theta_2=360.0;
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
  if (length <= 0)
    {
      warning("length-z should be positive, but is %g\n", length);
      return true;
    }
  if ((theta_1 < 0)||(theta_1>=360))
    {
      warning("initial theta should be positive or less than 360 deg., but is %g\n", theta_1);
      return true;
    }
  if ((theta_2 < 0)||(theta_2>360))
    {
      warning("final theta should be positive or less than 360 deg., but is %g\n", theta_2);
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
		     radius_y(radius_yv),
		     theta_1(0.0F),
		     theta_2(360.0F)

		    
{
  assert(length>0);
  assert(radius_x>0);
  assert(radius_y>0);
  this->set_origin(centre_v);
  if (this->set_direction_vectors(direction_vectors) == Succeeded::no)
    error("Ellipsoid constructor called with wrong direction_vectors");
}  

EllipsoidalCylinder::EllipsoidalCylinder(const float length_v, 
                         const float radius_xv,
	                 const float radius_yv,
	                 const float theta_1v,
	                 const float theta_2v,
	                 const CartesianCoordinate3D<float>& centre_v,
	                 const Array<2,float>& direction_vectors) 
                    :
                     length(length_v),
		     radius_x(radius_xv),
		     radius_y(radius_yv),
		     theta_1(theta_1v),
		     theta_2(theta_2v)

		    
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
      {
	const float x_pos=
	  r.x();
	const float y_pos=
	  r.y();
	const float phi_1 = 
	  static_cast<float>(_PI*theta_1/180.0);
	const float phi_2 = 
	  static_cast<float>(_PI*theta_2/180.0);
	// Adding theta cuts for partial cylinders from theta_1 to theta_2
	float theta_r = atan2(y_pos,x_pos);
	if (theta_r < 0.0) 
	  theta_r = static_cast<float>(2.0 * _PI + theta_r);
	return
	  (((phi_1 < phi_2)&&((theta_r >= phi_1)&&(theta_r <= phi_2))) ||
	   ((phi_1 > phi_2)&&((theta_r >= phi_1)||(theta_r <= phi_2))));
      }
    else 
      return false;
  }
  else return false;
}

float 
EllipsoidalCylinder:: 
get_geometric_volume()const
 {
   const float volume_of_unit_cell = this->get_volume_of_unit_cell();
   float A1=0, A2=0;
   float T1 = theta_1;
   float T2 = theta_2;
   if (theta_1 == theta_2)
     {
       T1 = 0.0;
       T2 = 360.0;
     }
   if (theta_1 == 360.0) T1 = 0.0;
   if (theta_2 ==   0.0) T2 = 360.0;
   const float phi_1 = static_cast<float>(T1*_PI/180.0);
   const float phi_2 = static_cast<float>(T2*_PI/180.0);
   
   // Begin Volume Calculation
   if (T1 == 0.0 && T2 == 360.0)
     return static_cast<float>(_PI*radius_x*radius_y*length/volume_of_unit_cell);
   if (radius_x == radius_y)
     {
       if (T1 < T2)
	 return static_cast<float>(_PI*radius_x*radius_y*length*
				   (T2-T1)/360.0/volume_of_unit_cell);
       else
	 return static_cast<float>(_PI*radius_x*radius_y*length*
				   (360.0-T1+T2)/360.0/volume_of_unit_cell);
     }
   
   if (T2 >=   0.0 && T2 <  90.0) //branch one
     A2 = atan2(radius_x/radius_y * fabs(tan(phi_2)),1);
   if (T2 >=  90.0 && T2 < 180.0) //branch two
     A2 = atan2(radius_x/radius_y * fabs(tan(phi_2)),-1);
   if (T2 >= 180.0 && T2 < 270.0) //branch three
     A2 = static_cast<float>(2*_PI + atan2(-radius_x/radius_y * fabs(tan(phi_2)),-1));
   if (T2 >= 270.0 && T2 <= 360.0) //branch four
     A2 = static_cast<float>(2*_PI + atan2(-radius_x/radius_y * fabs(tan(phi_2)),1));
   
   if (T1 >=   0.0 && T1 <  90.0) //branch one
     A1 = atan2(radius_x/radius_y * fabs(tan(phi_1)),1);
   if (T1 >=  90.0 && T1 < 180.0) //branch two
     A1 = atan2(radius_x/radius_y * fabs(tan(phi_1)),-1);
   if (T1 >= 180.0 && T1 < 270.0) //branch three
     A1 = static_cast<float>(2*_PI + atan2(-radius_x/radius_y * fabs(tan(phi_1)),-1));
   if (T1 >= 270.0 && T1 <= 360.0) //branch four
     A1 = static_cast<float>(2*_PI + atan2(-radius_x/radius_y * fabs(tan(phi_1)),1));
   
   if (T1 > T2)
     return static_cast<float>(radius_x*radius_y/2.0*
			       (2.0*_PI - A2 + A1)*length/volume_of_unit_cell);
   
   return static_cast<float>(radius_x*radius_y/2.0*(A2-A1)*length/volume_of_unit_cell);
 }

#if 0
//  scaling of axes does not simply scale area
float 
EllipsoidalCylinder:: 
get_geometric_area()const
{
  float T1 = theta_1, T2 = theta_2;
  if (theta_1 == theta_2)
    {
      T1 = 0.0;
      T2 = 360.0;
    }
  if (theta_1 == 360.0) T1 = 0.0;
  if (theta_2 ==   0.0) T2 = 360.0;
  const float phi_1 = T1*_PI/180.0;
  const float phi_2 = T2*_PI/180.0;
  
  // Begin Surface Area Calculation
  if (radius_x == radius_y)
    {
      if (theta_1 < theta_2)
	return static_cast<float>((2.0*_PI*radius_x*length + 2.0*_PI*radius_x*radius_y)
				  *(theta_2-theta_1)/360.0);
      else
	return static_cast<float>((2.0*_PI*radius_x*length + 2.0*_PI*radius_x*radius_y)
				  *(360.0-theta_1+theta_2)/360.0);
    }
  if (theta_1 == 0.0 && theta_2 == 360.0)
    {
      // Surface Area Uses Cantrell's approximation based upon 
      // Ramanujan's second approximation plus the area of the end caps
      float h = (radius_x - radius_y)/(radius_x + radius_y);
      h = h*h;
      float P = _PI*(radius_x + radius_y)*(1.0 + 3.0 * h/(10.0 + sqrt(4.0 - 3.0 * h))
					   + 4 * pow(h,12) * (1.0/_PI - 7.0/22.0));
      return static_cast<float>(P * length + 2.0 * _PI * radius_x * radius_y);
    }

  //  Simpson's Rule for an elliptic integral of the second kind 
  //  int(a*sqrt(1-e^2*sin^2(theta)),theta=phi_1...phi_2)  
  //  a = semi-major axis, b = semi-minor axis, e = eccentricty := 1-(b/a)^2, w/ b<=a
  const int n = 500;// n is arbitrailly set to ensure good accuracy
  const float dx = (phi_2 - phi_1)/static_cast<float>(n);
  float a = radius_x;
  float b = radius_y;
  
  if (radius_x < radius_y) 
    {
      a = radius_y;
      b = radius_x;
    }
  const float e2 = 1.0 - b*b/(a*a);
  
  float f = 0;
  f = sqrt(1.0 - e2 * sin(phi_1) * sin(phi_1));
  for (int j = 1; j <= n - 1 ; j = j + 2)
    f = f + 4.0 * sqrt(1.0 - e2 * sin(phi_1 + j * dx) * sin(phi_1 + j * dx));
  for (int j = 2; j <= n - 2 ; j = j + 2)
    f = f + 2.0 * sqrt(1.0 - e2 * sin(phi_1 + j * dx) * sin(phi_1 + j * dx));
  f = dx * a / 3.0 * (f + sqrt(1.0 - e2 * sin(phi_2) * sin(phi_2)));

  float r_1 = sqrt(radius_x*radius_x*(cos(phi_1)*cos(phi_1)) 
		   + radius_y*radius_y*(sin(phi_1)*sin(phi_1)));
  float r_2 = sqrt(radius_x*radius_x*(cos(phi_2)*cos(phi_2)) 
		   + radius_y*radius_y*(sin(phi_2)*sin(phi_2)));

  float area = get_geometric_volume()/length;
  
  return static_cast<float>((f+r_1+r_2)*length + 2.0*area);
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
    && std::fabs(theta_1 - cylinder.theta_1) < .1F
    && std::fabs(theta_2 - cylinder.theta_2) < .1F
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
