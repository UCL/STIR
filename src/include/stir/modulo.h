//
// $Id$
//
#ifndef __stir_modulo_H__
#define __stir_modulo_H__

/*!
  \file 
  \ingroup buildblock 
  \brief defines modulo() and related functions

  \author Kris Thielemans

  $Date$
  $Revision$

*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/common.h"

START_NAMESPACE_STIR

//! Like std::fmod() but with guaranteed nonnegative result
/*! \ingroup buildblock
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
/*!  \ingroup buildblock
  \see modulo(double,double)
  The reason for this function is that rounding from double to float
  might make the result of the calcluation with doubles larger than b.

  \warning Because of C++ promotion of floats to doubles, it is
  very easy to call the module(double,double) version inadvertently.
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
/*! \ingroup buildblock

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

//! A function to convert an angle from one range to another
/*! \ingroup buildblock
    This is mainly useful for converting results from e.g. std::atan2 to 
    a range \f$\[0,2\pi)\f$.
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
/*! \ingroup buildblock
    Identical to modulo(phi, 2*_PI) */
template <typename FloatOrDouble>
inline 
FloatOrDouble
to_0_2pi(const FloatOrDouble phi)
{
  return modulo(phi, static_cast<FloatOrDouble>(2*_PI));
}

END_NAMESPACE_STIR

#endif
