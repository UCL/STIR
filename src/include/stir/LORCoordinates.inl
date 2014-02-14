//
//

/*!
  \file 
  \ingroup LOR
  \brief Implementations for LORCoordinates.h
  \warning This is all preliminary and likely to change.
  \author Kris Thielemans


*/
/*
    Copyright (C) 2004- 2013, Hammersmith Imanet Ltd
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

#include "stir/modulo.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/********************************************************************
 The simple and boring constructors 
********************************************************************/
template <class coordT>
PointOnCylinder<coordT>::
PointOnCylinder()
  :_psi(0) // set to value to avoid assert
{}

template <class coordT>
PointOnCylinder<coordT>::
PointOnCylinder(const coordT z, const coordT psi)
  : _z(z), _psi(psi)
{}

template <class coordT>
LORInCylinderCoordinates<coordT>::
LORInCylinderCoordinates(const coordT radius)
  : _radius(radius)
{}

template <class coordT>
LORInCylinderCoordinates<coordT>::
LORInCylinderCoordinates(const PointOnCylinder<coordT>& p1,
			 const PointOnCylinder<coordT>& p2,
			 const coordT radius)
  : _p1(p1), _p2(p2), _radius(radius)
{}

template <class coordT>
LORAs2Points<coordT>::
LORAs2Points()
{}

template <class coordT>
LORAs2Points<coordT>::
LORAs2Points(const CartesianCoordinate3D<coordT>& p1,
	       const CartesianCoordinate3D<coordT>& p2)
  : _p1(p1), _p2(p2)
{}

template <class coordT>
LORInAxialAndSinogramCoordinates<coordT>::
LORInAxialAndSinogramCoordinates(const coordT radius)
  :  LORCylindricalCoordinates_z_and_radius<coordT>(radius),
     _phi(0),_s(0) // set _phi,_s to value to avoid assert
{
  check_state();
}

template <class coordT>
LORInAxialAndSinogramCoordinates<coordT>::
LORInAxialAndSinogramCoordinates(const coordT z1,
				 const coordT z2,
				 const coordT phi,
				 const coordT s,
				 const coordT radius)
  :
  LORCylindricalCoordinates_z_and_radius<coordT>(z1, z2, radius),
   _phi(phi), _s(s)
{
  check_state();
}

template <class coordT>
LORInAxialAndNoArcCorrSinogramCoordinates<coordT>::
LORInAxialAndNoArcCorrSinogramCoordinates(const coordT radius)
  : LORCylindricalCoordinates_z_and_radius<coordT>(radius),
   _phi(0),_beta(0)  // set _phi,_beta to value to avoid assert
  
{
  check_state();
}

template <class coordT>
LORInAxialAndNoArcCorrSinogramCoordinates<coordT>::
LORInAxialAndNoArcCorrSinogramCoordinates(const coordT z1,
				   const coordT z2,
				   const coordT phi,
				   const coordT beta,
				   const coordT radius)
  : LORCylindricalCoordinates_z_and_radius<coordT>(z1, z2, radius),
   _phi(phi), _beta(beta)
  
{
  check_state();
}

/********************************************************************
 more constructors
 These convert from one class to another
********************************************************************/
template <class coordT>
LORInCylinderCoordinates<coordT>::
LORInCylinderCoordinates(const LORInAxialAndNoArcCorrSinogramCoordinates<coordT>& na_coords)
  : _radius (na_coords.radius())
{
  _p1.z() = na_coords.z1();
  _p2.z() = na_coords.z2();
  _p1.psi() = to_0_2pi(na_coords.phi() + na_coords.beta());
  _p2.psi() = to_0_2pi(na_coords.phi() - na_coords.beta() + static_cast<coordT>(_PI));
  check_state();
}

template <class coordT>
LORInCylinderCoordinates<coordT>::
LORInCylinderCoordinates(const LORInAxialAndSinogramCoordinates<coordT>& coords)
  : _radius (coords.radius())
{
  _p1.z() = coords.z1();
  _p2.z() = coords.z2();
  _p1.psi() = to_0_2pi(coords.phi() + coords.beta());
  _p2.psi() = to_0_2pi(coords.phi() - coords.beta() + static_cast<coordT>(_PI));
  check_state();
}

template <class coordT>
static inline void 
get_sino_coords(  coordT& _z1,
		  coordT& _z2,
		  coordT& _phi,
		  coordT& _beta,
		  const LORInCylinderCoordinates<coordT>& cyl_coords)
{
  _beta = to_0_2pi((cyl_coords.p1().psi() - cyl_coords.p2().psi() +  static_cast<coordT>(_PI))/2);
  if (_beta>_PI)
    _beta -= static_cast<coordT>(2*_PI);
  _phi =  to_0_2pi((cyl_coords.p1().psi() + cyl_coords.p2().psi() - static_cast<coordT>( _PI))/2);

  /* beta is now between -Pi and Pi, phi between 0 and 2Pi
     Now bring into standard range and set z accordingly */
  if (_phi <  static_cast<coordT>(_PI))
    {
      if (_beta >=  static_cast<coordT>(_PI)/2)
        {
          _beta = static_cast<coordT>(_PI) - _beta;
	  _z2 = cyl_coords.p1().z();
	  _z1 = cyl_coords.p2().z();
        }
      else if (_beta <  -static_cast<coordT>(_PI)/2)
        {
          _beta = -static_cast<coordT>(_PI) - _beta;
	  _z2 = cyl_coords.p1().z();
	  _z1 = cyl_coords.p2().z();
        }
      else

	{
	  _z1 = cyl_coords.p1().z();
	  _z2 = cyl_coords.p2().z();
	}
    }
  else
    {
      _phi -= static_cast<coordT>(_PI);
      assert(_phi>=0);
      if (_beta >=  static_cast<coordT>(_PI)/2)
        {
          _beta -= static_cast<coordT>(_PI);
	  _z1 = cyl_coords.p1().z();
	  _z2 = cyl_coords.p2().z();
        }
      else if (_beta <  -static_cast<coordT>(_PI)/2)
        {
          _beta += static_cast<coordT>(_PI);
	  _z1 = cyl_coords.p1().z();
	  _z2 = cyl_coords.p2().z();
        }
      else
        {
          _beta *= -1;
	  _z2 = cyl_coords.p1().z();
	  _z1 = cyl_coords.p2().z();
        }
    }
  assert(_phi>=0);
  assert(_phi<=_PI);
  assert(_beta>=-_PI/2);
  assert(_beta<=_PI/2);    
}

template <class coordT>
LORInAxialAndNoArcCorrSinogramCoordinates<coordT>::
LORInAxialAndNoArcCorrSinogramCoordinates(const LORInCylinderCoordinates<coordT>& cyl_coords)
  :    LORCylindricalCoordinates_z_and_radius<coordT>(cyl_coords.radius())
{
#ifndef NDEBUG
  // set these to prevent assert breaking in check_state()
  _phi=0;
  _beta=0;
#endif
  get_sino_coords(z1(), z2(), _phi, _beta,
		  cyl_coords);
  check_state();
}

template <class coordT>
LORInAxialAndSinogramCoordinates<coordT>::
LORInAxialAndSinogramCoordinates(const LORInCylinderCoordinates<coordT>& cyl_coords)
  :    LORCylindricalCoordinates_z_and_radius<coordT>(cyl_coords.radius())
{
  coordT beta;
#ifndef NDEBUG
  // set these to prevent assert breaking in check_state()
  _phi=0;
  _s=0;
#endif
  get_sino_coords(z1(), z2(), _phi, beta,
		  cyl_coords);
  _s = this->_radius*sin(beta);
  check_state();
}

template <class coordT>
LORInAxialAndSinogramCoordinates<coordT>::
LORInAxialAndSinogramCoordinates(const LORInAxialAndNoArcCorrSinogramCoordinates<coordT>& coords)
  :    LORCylindricalCoordinates_z_and_radius<coordT>(coords.z1(), coords.z2(), coords.radius()),
       _phi(coords.phi()),
       _s(coords.s())
{
  check_state();
}

template <class coordT>
LORInAxialAndNoArcCorrSinogramCoordinates<coordT>::
LORInAxialAndNoArcCorrSinogramCoordinates(const LORInAxialAndSinogramCoordinates<coordT>& coords)
  :     LORCylindricalCoordinates_z_and_radius<coordT>(coords.z1(), coords.z2(), coords.radius()),
        _phi(coords.phi()),
        _beta(coords.beta())
{
  check_state();
}

#if __cplusplus>= 201103L
template <class coordT>
LORInAxialAndSinogramCoordinates<coordT>::
LORInAxialAndSinogramCoordinates(const LORAs2Points<coordT>& coords)
  : LORInAxialAndSinogramCoordinates(LORInCylinderCoordinates<coordT>(coords))
{}

template <class coordT>
LORInAxialAndNoArcCorrSinogramCoordinates<coordT>::
LORInAxialAndNoArcCorrSinogramCoordinates(const LORAs2Points<coordT>& coords)
  : LORInAxialAndNoArcCorrSinogramCoordinates(LORInCylinderCoordinates<coordT>(coords))
{}
#endif

template <class coordT>
LORAs2Points<coordT>::
LORAs2Points(const LORInCylinderCoordinates<coordT>& cyl_coords)
{
  // make sure the return values are in STIR coordinates
  p1().z() = cyl_coords.p1().z();
  p1().y() = -cyl_coords.radius()*cos(cyl_coords.p1().psi());
  p1().x() = cyl_coords.radius()*sin(cyl_coords.p1().psi());

  p2().z() = cyl_coords.p2().z();
  p2().y() = -cyl_coords.radius()*cos(cyl_coords.p2().psi());
  p2().x() = cyl_coords.radius()*sin(cyl_coords.p2().psi()); 
}

template <class coordT>
LORAs2Points<coordT>::
LORAs2Points(const LORInAxialAndSinogramCoordinates<coordT>& coords)
{
  *this = LORInCylinderCoordinates<coordT>(coords);
}

template <class coordT>
LORAs2Points<coordT>::
LORAs2Points(const LORInAxialAndNoArcCorrSinogramCoordinates<coordT>& coords)
{
  *this = LORInCylinderCoordinates<coordT>(coords);
}


template <class coordT1, class coordT2>
Succeeded
find_LOR_intersections_with_cylinder(LORAs2Points<coordT1>& intersection_coords,
				     const LORAs2Points<coordT2>& coords,
				     const double radius)
{
  const CartesianCoordinate3D<coordT2>& c1 = coords.p1();
  const CartesianCoordinate3D<coordT2>& c2 = coords.p2();

  const CartesianCoordinate3D<coordT2> d = c2 - c1;
  /* parametrisation of LOR is 
     c = l*d+c1
     l has to be such that c.x^2 + c.y^2 = R^2
     i.e.
     (l*d.x+c1.x)^2+(l*d.y+c1.y)^2==R^2
     l^2*(d.x^2+d.y^2) + 2*l*(d.x*c1.x + d.y*c1.y) + c1.x^2+c2.y^2-R^2==0
     write as a*l^2+2*b*l+e==0
     l = (-b +- sqrt(b^2-a*e))/a
     argument of sqrt simplifies to
     R^2*(d.x^2+d.y^2)-(d.x*c1.y-d.y*c1.x)^2
  */
  const double dxy2 = (square(d.x())+square(d.y()));
  const double argsqrt=
    (square(radius)*dxy2-square(d.x()*c1.y()-d.y()*c1.x()));
  if (argsqrt<=0)
    return Succeeded::no; // LOR is outside detector radius
  const coordT2 root = static_cast<coordT2>(sqrt(argsqrt));

  const coordT2 l1 = static_cast<coordT2>((- (d.x()*c1.x() + d.y()*c1.y())+root)/dxy2);
  const coordT2 l2 = static_cast<coordT2>((- (d.x()*c1.x() + d.y()*c1.y())-root)/dxy2);
  // TODO won't work when coordT1!=coordT2
  intersection_coords.p1() = d*l1 + c1;
  intersection_coords.p2() = d*l2 + c1;
  assert(fabs(square(intersection_coords.p1().x())+square(intersection_coords.p1().y())
	      - square(radius))
	 <square(radius)*10.E-5);
  assert(fabs(square(intersection_coords.p2().x())+square(intersection_coords.p2().y())
	      - square(radius))
	 <square(radius)*10.E-5);
  return Succeeded::yes;
}

template <class coordT1, class coordT2>
Succeeded
find_LOR_intersections_with_cylinder(LORInCylinderCoordinates<coordT1>& cyl_coords,
				     const LORAs2Points<coordT2>& cart_coords,
				     const double radius)
{
  LORAs2Points<coordT1> intersection_coords;
  if (find_LOR_intersections_with_cylinder(intersection_coords, cart_coords, radius)
      == Succeeded::no)
    return Succeeded::no;

  const CartesianCoordinate3D<coordT1>& c1 = intersection_coords.p1();
  const CartesianCoordinate3D<coordT1>& c2 = intersection_coords.p2();
  cyl_coords.reset(static_cast<float>(radius));
  
  cyl_coords.p1().psi() = 
    from_min_pi_plus_pi_to_0_2pi(static_cast<coordT1>(atan2(c1.x(),-c1.y())));
  cyl_coords.p2().psi() = 
    from_min_pi_plus_pi_to_0_2pi(static_cast<coordT1>(atan2(c2.x(),-c2.y())));
  cyl_coords.p1().z() = 
    static_cast<coordT1>(c1.z());
  cyl_coords.p2().z() = 
    static_cast<coordT1>(c2.z());

  return Succeeded::yes;
}

template <class coordT1, class coordT2>
Succeeded
find_LOR_intersections_with_cylinder(LORInAxialAndNoArcCorrSinogramCoordinates<coordT1>&  lor,
				     const LORAs2Points<coordT2>& cart_coords,
				     const double radius)
{
  LORInCylinderCoordinates<coordT1> cyl_coords;
  if (find_LOR_intersections_with_cylinder(cyl_coords, cart_coords, radius)
      == Succeeded::no)
    return Succeeded::no;
  lor = cyl_coords;
  return Succeeded::yes;
}


template <class coordT1, class coordT2>
Succeeded
find_LOR_intersections_with_cylinder(LORInAxialAndSinogramCoordinates<coordT1>&  lor,
				     const LORAs2Points<coordT2>& cart_coords,
				     const double radius)
{
  LORInCylinderCoordinates<coordT1> cyl_coords;
  if (find_LOR_intersections_with_cylinder(cyl_coords, cart_coords, radius)
      == Succeeded::no)
    return Succeeded::no;
  lor = cyl_coords;
  return Succeeded::yes;
}


template <class coordT>
Succeeded
LORAs2Points<coordT>::
change_representation(LORInCylinderCoordinates<coordT>& lor,
		   const double radius) const
{
  return find_LOR_intersections_with_cylinder(lor, *this, radius);
}

template <class coordT>
Succeeded
LORAs2Points<coordT>::
change_representation(LORInAxialAndNoArcCorrSinogramCoordinates<coordT>& lor,
		   const double radius) const
{
  return find_LOR_intersections_with_cylinder(lor, *this, radius);
}

template <class coordT>
Succeeded
LORAs2Points<coordT>::
change_representation(LORInAxialAndSinogramCoordinates<coordT>& lor,
		   const double radius) const
{
  return find_LOR_intersections_with_cylinder(lor, *this, radius);
}

template <class coordT>
Succeeded
LORAs2Points<coordT>::
get_intersections_with_cylinder(LORAs2Points<coordT>& lor,
                                const double radius) const
{
  return find_LOR_intersections_with_cylinder(lor, *this, radius);
}

#define DEFINE_LOR_GET_FUNCTIONS(TYPE)                                       \
template <class coordT>							     \
Succeeded					     \
TYPE<coordT>::								     \
change_representation(LORInCylinderCoordinates<coordT>& lor,              \
                         const double radius)  const	  	             \
{								             \
  lor = *this;                           				     \
  return lor.set_radius(static_cast<coordT>(radius));					     \
}									     \
									     \
template <class coordT>							     \
Succeeded			     \
TYPE<coordT>::								     \
change_representation(LORInAxialAndNoArcCorrSinogramCoordinates<coordT>& lor,              \
                                               const double radius)  const   \
{									     \
  lor = *this;                           				     \
  return lor.set_radius(static_cast<coordT>(radius));					     \
}									     \
									     \
template <class coordT>							     \
Succeeded				     \
TYPE<coordT>::								     \
change_representation(LORInAxialAndSinogramCoordinates<coordT>& lor,              \
                                  const double radius) const                 \
{									     \
  lor = *this;                           				     \
  return lor.set_radius(static_cast<coordT>(radius));					     \
}									     \
									     \
template <class coordT>							     \
Succeeded							     \
TYPE<coordT>::								     \
get_intersections_with_cylinder(LORAs2Points<coordT>& lor,              \
                                const double radius) const		     \
{									     \
  self_type tmp = *this;							     \
  if (tmp.set_radius(static_cast<coordT>(radius)) == Succeeded::no)			             \
    return Succeeded::no;						     \
  lor = tmp;                                                                 \
  return Succeeded::yes;								     \
}
									    
DEFINE_LOR_GET_FUNCTIONS(LORInCylinderCoordinates)			    
DEFINE_LOR_GET_FUNCTIONS(LORInAxialAndNoArcCorrSinogramCoordinates)
DEFINE_LOR_GET_FUNCTIONS(LORInAxialAndSinogramCoordinates)

#undef DEFINE_LOR_GET_FUNCTIONS


									     
									     
END_NAMESPACE_STIR							     
