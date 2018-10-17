//
//

/*! 
  \file 
  \ingroup geometry
  \brief  A few functions to compute distances between lines etc
  \todo move implementations to .cxx
  \author Nikos Efthimiou
  \author Kris Thielemans
*/
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
    Copyright (C) 2016, University of Hull

    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details.
*/

#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"
#include <cmath>
#ifdef BOOST_NO_STDC_NAMESPACE
  namespace std { using ::sqrt; using ::fabs; }
#endif


START_NAMESPACE_STIR

/*! \ingroup geometry
  \brief find a point half-way between 2 lines and their distance

  The point is found by minimising the distance with both lines.

  If the lines are parallel, the point returned is still valid, although it is of course 
  not unique in practice.
*/
template <class coordT>
inline 
coordT
coordinate_between_2_lines(CartesianCoordinate3D<coordT>& result,
			   const LORAs2Points<coordT>& line0, 
			   const LORAs2Points<coordT>& line1)
{
  /* Rationale:
     parametrise points on the lines as
       r0 + a0 d0, r1 + a1 d1
     where r0 is a point on the line, and d0 is its direction

     compute distance squared between those points, i.e.
       norm(a0 d0 - a1 d1 + r0-r1)^2
     This can be written using inner products of all the vectors.

     Minimising it (by computing derivatives) results in a linear eq in a0, a1:
        a0 d0d0 == a1 d0d1 + r10d0, 
	a0 d0d1 == a1 d1d1 + r10d1
     which is easily solved.
     The half-way point can then be found by using 
        (r0 + a0 d0 + r1 + a1 d1)/2
     for the a0,a1 found.
  */
  const CartesianCoordinate3D<coordT>& r0 = line0.p1();
  const CartesianCoordinate3D<coordT>& r1 = line1.p1();
  const CartesianCoordinate3D<coordT> r10 = r1-r0;
  const CartesianCoordinate3D<coordT> d0 = line0.p2() - line0.p1();
  const CartesianCoordinate3D<coordT> d1 = line1.p2() - line1.p1();
  const coordT d0d0 = inner_product(d0,d0);
  const coordT d0d1 = inner_product(d0,d1);
  const coordT d1d1 = inner_product(d1,d1);
  const coordT r10d0 = inner_product(r10,d0);
  const coordT r10r10 = inner_product(r10,r10);

  const coordT eps=d0d0*10E-5; // small number for comparisons

  const coordT denom = square(d0d1) - d0d0*d1d1;
  coordT distance_squared;
  if (std::fabs(denom) <= eps)
    {
      //parallel lines
      const coordT a0 = r10d0/d0d0;
      result = r0 + d0*a0;
      distance_squared = r10r10 - square(r10d0)/d0d0;
    }
  else
    {
      const coordT r10d1 = inner_product(r10,d1);
      const coordT a0 = (-d1d1*r10d0 + d0d1*r10d1)/denom;
      const coordT a1 = (-d0d1*r10d0 + d0d0*r10d1)/denom;
      
      result = ((r0 + d0*a0) + (r1 + d1*a1))/2;
      distance_squared =
	(d1d1*square(r10d0) - 2*d0d1*r10d0*r10d1 + d0d0*square(r10d1))/denom + 
	r10r10;
    }
  if (distance_squared >= 0)
    return std::sqrt(distance_squared);
  else
    {
      if (-distance_squared < eps)
	return 0;
      else
	{
	  assert(false);
	  return std::sqrt(distance_squared); // will return NaN 
	}
    }
}


/*! \ingroup geometry
  \brief find the distance between a point and a line

*/
template <class coordT>
inline
coordT
distance_between_line_and_point(
				const LORAs2Points<coordT>& line, 
				const CartesianCoordinate3D<coordT>& r1 )
{
  /* Rationale:
     parametrise points on the lines as
       r0 + a0 d0
     where r0 is a point on the line, and d0 is its direction

     compute distance squared between those points, i.e.
       norm(a0 d0 + r0-r1)^2
     This can be written using inner products of all the vectors:
       a0^2 d0d0 - 2 a0 r10d0 + r10r10
     Minimising it (by computing derivatives) results in a linear eq in a0, a1:
        a0 d0d0 == r10d0
  */  
  const CartesianCoordinate3D<coordT>& r0 = line.p1();
  const CartesianCoordinate3D<coordT> r10 = r1-r0;
  const CartesianCoordinate3D<coordT> d0 = line.p2() - line.p1();
  const coordT d0d0 = inner_product(d0,d0);
  const coordT r10d0 = inner_product(r10,d0);
  const coordT r10r10 = inner_product(r10,r10);

  //const coordT a0 = r10d0/d0d0;
  //result = r0 + d0*a0;
  const coordT distance_squared = r10r10 - square(r10d0)/d0d0;
  if (distance_squared >= 0)
    return std::sqrt(distance_squared);
  else
    {
      if (-distance_squared < d0d0*10E-5)
	return 0;
      else
	{
	  assert(false);
	  return std::sqrt(distance_squared); // will return NaN
	}
    }
}

/*! \ingroup geometry
  \brief Project a point on a line.

  \author Nikos Efthimiou
*/
template <class coordT>
inline void
project_point_on_a_line(
        const CartesianCoordinate3D<coordT>& p1,
        const CartesianCoordinate3D<coordT>& p2,
        CartesianCoordinate3D<coordT>& r1 )
{

    const CartesianCoordinate3D<coordT> difference = p2 - p1;

    const CartesianCoordinate3D<coordT> r10 = r1 - p1;

    float inner_prod = inner_product(difference, difference);

    const float u = inner_product(r10, difference) / inner_prod ;

    r1.x() = p1.x() + u * difference.x();
    r1.y() = p1.y() + u * difference.y();
    r1.z() = p1.z() + u * difference.z();

}

template <class coordT>
inline void
project_point_on_a_line2(
        const CartesianCoordinate3D<coordT>& p1,
        const CartesianCoordinate3D<coordT>& p2,
        CartesianCoordinate3D<coordT>& r1,
        bool& sign)
{

    const CartesianCoordinate3D<coordT> difference = p2 - p1;

    const CartesianCoordinate3D<coordT> r10 = r1 - p1;

    const float u = inner_product(r10, difference) /
            inner_product(difference, difference);

    r1[3] = u * difference[3];
    r1[2] = u * difference[2];
    r1[1] = u * difference[1];

    sign = u > 0 ? true : false;
}

END_NAMESPACE_STIR
