//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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

/*!
  \file 
  \ingroup numerics
  \brief Implementation of the stir::norm(), stir::norm_squared() functions and 
  stir::NormSquared unary function

  \author Kris Thielemans

  $Date$
  $Revision$
*/


#include <functional>
#include <cmath>
# ifdef BOOST_NO_STDC_NAMESPACE
 namespace std { using ::fabs; }
# endif

START_NAMESPACE_STIR


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


template <class Iter>
double norm_squared (Iter begin, Iter end)
{ 
  double res = 0;
  for (Iter iter= begin; iter != end; ++iter)
      res+= norm_squared(*iter);
  return res;
}

template <class Iter>
double norm (Iter begin, Iter end)
{ 
  return sqrt(norm_squared(begin, end));
}


template<class elemT>
inline double 
norm (const Array<1,elemT> & v1)
{
  return norm(v1.begin(), v1.end());
}

template<class elemT>
inline double 
norm_squared(const Array<1,elemT> & v1)
{
  return norm_squared(v1.begin(), v1.end());
}

END_NAMESPACE_STIR
#endif
