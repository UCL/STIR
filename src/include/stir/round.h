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
    This software is distributed under the terms of the GNU Lesser General 
    Public Licence (LGPL).
    See STIR/LICENSE.txt for details
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
   The usual <tt>(int)(x+.5)</tt> has machine dependent behaviour for 
   negative numbers.

   .5 is rounded to 1 (and hence -.5 to -1).

   \warning There is no check on overflow (i.e. if x is too 
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

   \warning if an unsigned type is used for \x integerT, the result for negative \a x
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
