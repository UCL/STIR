
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
//@}

END_NAMESPACE_STIR

#endif
