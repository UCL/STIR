//
//
#ifndef __stir_modulo_H__
#define __stir_modulo_H__

/*!
  \file 
  \ingroup buildblock 
  \brief defines stir::modulo() and related functions

  \author Kris Thielemans


*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
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

#include "stir/common.h"

START_NAMESPACE_STIR

/*! \ingroup buildblock
   \name Functions for modulo computations
*/
//@{

//! Like std::fmod() but with guaranteed nonnegative result
/*! 
    std::fmod(a,b) return a number of the same sign as \a a. This is often 
    inconvenient as the result of this is that the range of std::fmod(a,b) is
    from \a -fabs(b) to \a +fabs(b).

    In contrast, modulo(a,b) always returns a nonnegative result. To be precise:

    \c modulo(a,b) returns the value \c a-i*b, for the integer \c i 
    such  that, if \c b is nonzero, the result is greater or equal to 0 and
    less than the magnitude of \c b.

    Error handling (i.e. \a b=0) is as for std::fmod().

    \warning When assigning the result to a float, the implied rounding might
   give you float which is (a tiny bit) larger than fabs(b).

*/
inline
double
modulo(const double a, const double b)
{
  const double res = fmod(a,b);
  return res<0 ? res + fabs(b) : res;
}

//! modulo for floats
/*!  
  \see modulo(double,double)
  The reason for this function is that rounding from double to float
  might make the result of the calculation with doubles larger than b.

  \warning Because of C++ promotion of floats to doubles, it is
  very easy to call the modulo(double,double) version inadvertently.
  So, you should probably not rely too much on the result being less than 
  \a b.
*/
inline
float
modulo(const float a, const float b)
{
  float res = 
    static_cast<float>(modulo(static_cast<double>(a),static_cast<double>(b)));
  assert(res>=0);
  const float abs_b = b>=0 ? b : -b;
  if (res>=abs_b) res -= abs_b;
  assert(res>=0);
  return res;
}

//! Like the normal modulus operator (%) but with guaranteed nonnegative result
/*! 
   Result will be larger than or equal to 0, and (strictly) smaller than
   \a abs(b).
*/
inline
int
modulo(const int a, const int b)
{
  const int res = a%b;
  const int res2 = res<0 ? res + (b>=0 ? b : -b) : res;
  assert(res2>=0);
  assert(res2<(b>=0?b:-b));
  return res2;
}

//! Performs the modulus operation on each element of the coordinates
/*! 
   \return A BasicCoordinate such that for all <tt>d</tt>
     \code result[d] = modulo(a[d], b[d] \endcode
*/
template <int num_dimensions, typename T>
inline
BasicCoordinate<num_dimensions, T>
modulo(const BasicCoordinate<num_dimensions, T>& a, const BasicCoordinate<num_dimensions, T>& b)
{
  BasicCoordinate<num_dimensions, T> result;
  for (int d=1; d<=num_dimensions; ++d)
    result[d] = modulo(a[d], b[d]);
  return result;
}

//! A function to convert an angle from one range to another
/*! 
    This is mainly useful for converting results from e.g. std::atan2 to 
    a range \f$[0,2\pi)\f$.
*/
template <typename FloatOrDouble>
inline 
FloatOrDouble
from_min_pi_plus_pi_to_0_2pi(const FloatOrDouble phi)
{
  static const FloatOrDouble two_pi =static_cast<FloatOrDouble>(2*_PI);
  assert(phi>= -two_pi);
  assert(phi< two_pi);
  if (phi>=0)
    return phi;
  FloatOrDouble res = phi+two_pi;
  // due to floating point finite precision, the following check is needed...
  if (res>=two_pi)
      res -= two_pi;
  assert(res>=0);
  assert(res<two_pi);
  return res;
}

//! Convert angle to standard range
/*! Identical to <tt>modulo(phi, static_cast<FloatOrDouble>(2*_PI))</tt> */
template <typename FloatOrDouble>
inline 
FloatOrDouble
to_0_2pi(const FloatOrDouble phi)
{
  return modulo(phi, static_cast<FloatOrDouble>(2*_PI));
}

//! Convert angle to standard range when period is 180 degrees
/*! Identical to <tt>modulo(phi, static_cast<FloatOrDouble>(_PI))</tt> */
template <typename FloatOrDouble>
inline
FloatOrDouble
to_0_pi(const FloatOrDouble phi)
{
  return modulo(phi, static_cast<FloatOrDouble>(_PI));
}

//@}

END_NAMESPACE_STIR

#endif
