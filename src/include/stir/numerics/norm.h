//
//
/*
    Copyright (C) 2003- 2005, Hammersmith Imanet Ltd
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

#ifndef __stir_numerics_norm_H__
#define __stir_numerics_norm_H__
/*!
  \file 
  \ingroup numerics
  \brief Declaration of the stir::norm(), stir::norm_squared() functions and 
  stir::NormSquared unary function

  \author Kris Thielemans

*/


#include "stir/common.h"
#include <complex>
#include <cmath>
# ifdef BOOST_NO_STDC_NAMESPACE
 namespace std { using ::fabs; }
# endif

START_NAMESPACE_STIR

template <int num_dimensions, class elemT> class Array;

/*!
 \ingroup numerics
 \name Functions to compute the norm
*/
//@{

//! A helper class that computes the square of the norm of numeric data
/*! It's there to avoid unnecessary sqrts when computing norms of vectors.
    
    The default template just uses the square (with conversion to double), 
    but we'll specialise it for complex numbers.

    It's derived from std::unary_function such that it follows
    the conventions for a function object.
*/    

// specialisations for complex numbers are in .inl file for clarity of this file.
template <typename T>
struct NormSquared :
  public std::unary_function<T, double>
{
  double operator()(T x) const
  {   
    return static_cast<double>(x)*x;
  }
};


//! Returns the square of the norm of a number
/*! \see norm(elemT)*/
template <typename elemT>
inline double
norm_squared (const elemT t)
{ return NormSquared<elemT>()(t); }


//! Returns the norm of a number
/*! This is the same as the absolute value, but works also for std::complex<T>.*/
template <typename elemT>
inline double
norm (const elemT t)
{ return sqrt(norm_squared(t)); }

// 2 overloads to avoid doing sqrt(t*t)
inline double
norm (const double t)
{ return std::fabs(t); }

inline double
norm (const float t)
{ return std::fabs(t); }

//! Returns the square of the l2-norm of a sequence
/*! The l2-norm is defined as the sqrt of the sum of squares of the norms
    of each element in the sequence.
*/
template <class Iter>
inline double norm_squared (Iter begin, Iter end);

//! Returns the l2-norm of a sequence
/*! The l2-norm is defined as the sqrt of the sum of squares of the norms
    of each element in the sequence.

    \see norm(const Array<1,elemT>&) for a convenience function for Array objects.
*/
template <class Iter>
inline double norm (Iter begin, Iter end);

//! l2 norm of a 1D array
/*! 
  This returns the sqrt of the sum of the square of the absolute values 
  of the elements of \a v1.
 */
template<class elemT>
inline double 
norm (const Array<1,elemT> & v1);


//! square of the l2 norm of a 1D array
/*! 
  This returns the sum of the square of the absolute values of the  
  elements of \a v1.
 */
template<class elemT>
inline double 
norm_squared (const Array<1,elemT> & v1);

//@}

END_NAMESPACE_STIR

#include "stir/numerics/norm.inl"

#endif
