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
#ifndef __stir_round_H__
#define __stir_round_H__
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of the stir::round functions
    
  \author Kris Thielemans
  \author Charalampos Tsoumpas
      
  $Date$
  $Revision$
*/
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR
/*!
 \ingroup buildblock
 \name Functions for rounding floating point numbers
 */
 //@{

//! Implements rounding of floating point numbers
/*!
   
   round() has the property that 
   \code
   round(x) == -round(-x)
   \endcode
   The usual <code>(int)(x+.5)</code> has machine dependent behaviour for 
   negative numbers.

   .5 is rounded to 1 (and hence -.5 to -1).

   \warning There is no check on overflow (i.e. if \a x is too 
   large to fit in an \c int).
*/
inline int round(const float x);
//! Implements rounding of double numbers
/*!   
   \see round(const float)
*/
inline int round(const double x);

//! Implements rounding of a BasicCoordinate object
/*!
   \see round(const float)
*/
template <int num_dimensions, class elemT>
inline BasicCoordinate<num_dimensions,int>
round(const BasicCoordinate<num_dimensions,elemT>& x);


//! Implements rounding of floating point numbers to other integer types
/*!
   This function is templated in the output type for convenience of implementation.

   \warning if an unsigned type is used for \c integerT, the result for negative \a x
   will be system dependent

   \see round(const float)

   \todo add code to check that \c integerT is really an integer type at compilation time
 */
template <typename integerT>
inline void
round_to(integerT& result, const float x);

//! Implements rounding of a BasicCoordinate object to other integer types
/*! \see round_to(integerT, float)
 */
template <int num_dimensions, class integerT, class elemT>
inline void
round_to(BasicCoordinate<num_dimensions,integerT>& result,
	 const BasicCoordinate<num_dimensions,elemT>& x);

//@}

END_NAMESPACE_STIR

#include "stir/round.inl"

#endif
