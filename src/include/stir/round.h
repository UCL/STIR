//
// $Id$
//
#ifndef __stir_round_H__
#define __stir_round_H__
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of the round functions
    
  \author Kris Thielemans
  \author Charalampos Tsoumpas
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

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

//! Implements rounding of a BasicCoordinate object
/*!
   \ingroup buildblock
   \see round(const float)
*/
template <int num_dimensions, class elemT>
inline BasicCoordinate<num_dimensions,int>
round(const BasicCoordinate<num_dimensions,elemT>& x);

END_NAMESPACE_STIR

#include "stir/round.inl"

#endif
