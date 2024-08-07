
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_assign_H__
#define __stir_assign_H__

/*!
  \file
  \ingroup buildblock
  \brief defines the stir::assign function to assign values to different data types

  \author Kris Thielemans

*/
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include <vector>

START_NAMESPACE_STIR
/*! \ingroup buildblock
  \name templated functions for assigning values

  When writing templated code, it is sometimes not possible to use \c operator=()
  for assignment, e.g. when the classes do not support that operator. The
  \c assign template tries to alleviate this problem by providing several
  overloads when the first argument is a (STIR) container.

  \par Usage
  \code
  assign(x,y); // logically equivalent to x=y;
  \endcode

  \par Design consideration
  We could have overloaded \c operator=() instead, but that would probably
  lead to surprising conversions.
*/
//@{

template <class T, class T2>
inline void
assign(T& x, const T2& y)
{
  x = y;
}

template <class T, class T2>
inline void
assign(std::vector<T>& v, const T2& y)
{
  for (typename std::vector<T>::iterator iter = v.begin(); iter != v.end(); ++iter)
    assign(*iter, y);
}

template <int num_dimensions, class T, class T2>
inline void
assign(BasicCoordinate<num_dimensions, T>& v, const T2& y)
{
  for (typename BasicCoordinate<num_dimensions, T>::iterator iter = v.begin(); iter != v.end(); ++iter)
    assign(*iter, y);
}

template <class T, class T2>
inline void
assign(VectorWithOffset<T>& v, const T2& y)
{
  for (typename VectorWithOffset<T>::iterator iter = v.begin(); iter != v.end(); ++iter)
    assign(*iter, y);
}

// Even though we have VectorWithOffset above, we still seem to need a version for Arrays as well
// for when calling assign(vector<array<1,float> >, 0).
// We're not sure why...
template <int num_dimensions, class T, class T2>
inline void
assign(Array<num_dimensions, T>& v, const T2& y)
{
  for (typename Array<num_dimensions, T>::full_iterator iter = v.begin_all(); iter != v.end_all(); ++iter)
    assign(*iter, y);
}

// a few common cases given explictly here such that we don't get conversion warnings all the time.
inline void
assign(double& x, const int y)
{
  x = static_cast<double>(y);
}

inline void
assign(float& x, const int y)
{
  x = static_cast<float>(y);
}
//@}

END_NAMESPACE_STIR

#endif
