
// $Id$

#ifndef __Array_complex_numbers_H__
#define __Array_complex_numbers_H__

/*!
  \file 
  \ingroup Array 
  \brief defines additional numerical operations for arrays of complex numbers

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
#include "stir/Array.h"
#include <complex>

START_NAMESPACE_STIR

/*! @name numerical operations for arrays of complex numbers
 \ingroup Array
*/
//@{

template <int num_dimensions, typename elemT>
Array<num_dimensions, std::complex<elemT> >&
operator *= (Array<num_dimensions, std::complex<elemT> >& lhs, const elemT& rhs)
{
  typename Array<num_dimensions, std::complex<elemT> >::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ *= rhs;
  return lhs;
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, std::complex<elemT> >&
operator /= (Array<num_dimensions, std::complex<elemT> >& lhs, const elemT& rhs)
{
  typename Array<num_dimensions, std::complex<elemT> >::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ /= rhs;
  return lhs;
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, std::complex<elemT> >&
operator += (Array<num_dimensions, std::complex<elemT> >& lhs, const elemT& rhs)
{
  typename Array<num_dimensions, std::complex<elemT> >::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ += rhs;
  return lhs;
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, std::complex<elemT> >&
operator -= (Array<num_dimensions, std::complex<elemT> >& lhs, const elemT& rhs)
{
  typename Array<num_dimensions, std::complex<elemT> >::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ -= rhs;
  return lhs;
}


// a few common cases given explictly here such that we don't get conversion warnings all the time.
inline 
void assign(std::complex<double>& x, const int y)
{
  x=static_cast<std::complex<double> >(y);
}

inline 
void assign(std::complex<float>& x, const int y)
{
  x=static_cast<float>(y);
}
//@}
END_NAMESPACE_STIR

#endif
