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
    from \a -b to \a +b.

    In contrast, modulo(a,b) always returns a nonnegative result. To be precise:

    \c modulo(a,b) returns the value \c a-i*b, for the integer \c i 
    such  that, if \c b is nonzero, the result is greater or equal to 0 and
    less than the magnitude of \c b.

    Error handling (i.e. \a b=0) is as for std::fmod().
*/
double
modulo(const double a, const double b)
{
  double res = fmod(a,b);
  return res<0 ? res + b : res;
}

//! A function to convert an angle from one range to another
/*! \ingroup bulidblock
    This is mainly useful for converting results from e.g. std::atan2 to 
    a range \f$\[0,2\pi)\f$.
*/
double
from_min_pi_plus_pi_to_0_2pi(const double phi)
{
  assert(phi>= -2*_PI);
  assert(phi< 2*_PI);
  return 
    phi<0? phi+2*_PI : phi;
}

//! Convert angle to standard range
/*! \ingroup bulidblock
    Identical to modulo(phi, 2*_PI) */
double
to_0_2pi(const double phi)
{
  return modulo(phi, 2*_PI);
}

END_NAMESPACE_STIR

#endif
