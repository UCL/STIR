//
// $Id$
//

#ifndef __stir_norm_H__
#define __stir_norm_H__

/*!
  \file 
  \ingroup buildblock
  \brief defines the norm() function and NormSquared unary function

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/common.h"
#include <complex>
#include <functional>

START_NAMESPACE_STIR

//! A helper class that computes the square of the norm of numeric data
/*! It's there to avoid unnecessary sqrts when computing norms of vectors.
    
    The default template just uses the square (with conversion to double), 
    but we'll specialise it for complex numbers.

    It's derived from std::unary_function such that it follows
    the conventions for a function object.
*/    
template <typename T>
struct NormSquared :
  public std::unary_function<T, double>
{
  double operator()(T x) const
  {   
    return static_cast<double>(x)*x;
  }
};

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T>
struct NormSquared<std::complex<T> >
  :   public std::unary_function<const std::complex<T>&, double>
{
  double operator()(const std::complex<T>& x) const
  { return square(x.real())+ square(x.imag()); }
};

#else //  BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

// handle float and double separately. (sigh)
#define INSTANTIATE_NormSquared(T)                              \
  template <>                                                    \
  struct NormSquared<std::complex<T> >                          \
  :   public std::unary_function<const std::complex<T>&, double> \
{                                                                \
  double operator()(const std::complex<T>& x) const              \
  { return square(x.real())+ square(x.imag()); }                 \
};
INSTANTIATE_NormSquared(float)
INSTANTIATE_NormSquared(double)
#undef INSTANTIATE_NormSquared
#endif

//! Returns the square of the norm of a number
/*! \see norm(elemT)*/
template <typename elemT>
double
norm_squared (const elemT t)
{ return NormSquared<elemT>()(t); }


//! Returns the norm of a number
/*! This is the same as the absolute value, but works also for std::complex<T>.*/
template <typename elemT>
double
norm (const elemT t)
{ return sqrt(norm_squared(t)); }

//! Returns the l2-norm of a sequence
/*! The l2-norm is defined as the sqrt of the sum of squares of the norms
    of each element in the sequence.
*/
template <class Iter>
double norm (Iter begin, Iter end)
{ 
  double res = 0;
  for (Iter iter= begin; iter != end; ++iter)
      res+= norm_squared(*iter);
  return sqrt(res);
}

END_NAMESPACE_STIR
#endif
