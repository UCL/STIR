
/*
    Copyright (C) 2005- 2008, Hammersmith Imanet Ltd
    Copyright (C) 2025, University College London
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
#include "stir/type_traits.h"
#include <typeinfo>

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

// generic implementation, used whenever there is no specialisation
// Note that std::enable_if_t without 2nd argument defaults to `void` (if the first argument is true, of course)
template <class T, class T2>
std::enable_if_t<!has_iterator_v<T>>
assign(T& x, const T2& y)
{
  x = y;
}

// implementation when the first argument has a (STIR) full iterator, e.g. Array
template <class T, class T2>
std::enable_if_t<has_full_iterator_v<T>>
assign(T& v, const T2& y)
{
  for (auto iter = v.begin_all(); iter != v.end_all(); ++iter)
    assign(*iter, y);
}

// implementation for normal iterators
template <class T, class T2>
std::enable_if_t<has_iterator_and_no_full_iterator<T>::value>
assign(T& v, const T2& y)
{
  for (auto& i : v)
    assign(i, y);
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
