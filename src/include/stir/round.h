//
// $Id$
//
#ifndef __Tomo_round_H__
#define __Tomo_round_H__
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of the round functions
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
#include "tomo/common.h"

START_NAMESPACE_TOMO

//! Implements rounding of floating point numbers
/*!
   \ingroup buildblock
   round() has the property that 
   \code
   round(x) == -round(-x)
   \endcode
   The usual (int)(x+.5) has machine dependent behaviour for 
   negative numbers.

   .5 is rounded to 1 (and hence -.5 to -1).

   \warning There is no check on overflow (i.e. if x is too 
   large to fit in an \c int).
*/
inline int round(const float x);
//! Implements rounding of double numbers
/*!
   \ingroup buildblock
   \see round(const float)
*/
inline int round(const double x);

END_NAMESPACE_TOMO

#include "tomo/round.inl"

#endif
