//
//
/*
    Copyright (C) 2004- 2008, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array

  \brief implementation of functions in stir/array_index_functions.h

  \author Kris Thielemans

*/
/*
    Copyright (C) 2004- 2008, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/detail/test_if_1d.h"
#include <algorithm>

START_NAMESPACE_STIR

/* First we define the functions that actually do the work.

  The code is a bit more complicated than need be because we try to accomodate
   older compilers that have trouble with function overloading of templates.
   See test_if_1d for some info.
*/
namespace detail
{

template <int num_dimensions, typename T>
inline BasicCoordinate<num_dimensions, int>
get_min_indices_help(is_not_1d, const Array<num_dimensions, T>& a)
{
  if (a.get_min_index() <= a.get_max_index())
    return join(a.get_min_index(), get_min_indices(*a.begin()));
  else
    {
      // a is empty.  Not clear what to return, so we just return 0
      // It would be better to throw an exception.
      BasicCoordinate<num_dimensions, int> tmp(0);
      return tmp;
    }
}

template <typename T>
inline BasicCoordinate<1, int>
get_min_indices_help(is_1d, const Array<1, T>& a)
{
  BasicCoordinate<1, int> result;
  result[1] = a.get_min_index();
  return result;
}

template <int num_dimensions2, typename T>
inline bool
next_help(is_1d, BasicCoordinate<1, int>& index, const Array<num_dimensions2, T>& a)
{
  if (a.get_min_index() > a.get_max_index())
    return false;
  assert(index[1] >= a.get_min_index());
  assert(index[1] <= a.get_max_index());
  index[1]++;
  return index[1] <= a.get_max_index();
}

template <typename T, int num_dimensions, int num_dimensions2>
inline bool
next_help(is_not_1d, BasicCoordinate<num_dimensions, int>& index, const Array<num_dimensions2, T>& a)
{
  if (a.get_min_index() > a.get_max_index())
    return false;
  BasicCoordinate<num_dimensions - 1, int> upper_index = cut_last_dimension(index);
  assert(index[num_dimensions] >= get(a, upper_index).get_min_index());
  assert(index[num_dimensions] <= get(a, upper_index).get_max_index());
  index[num_dimensions]++;
  if (index[num_dimensions] <= get(a, upper_index).get_max_index())
    return true;
  if (!next(upper_index, a))
    return false;
  index = join(upper_index, get(a, upper_index).get_min_index());
  return true;
}

} // end of namespace detail

/* Now define the functions in the stir namespace in terms of the above.
   Also define get() for which I didn't bother to try the work-arounds,
   as they don't work for VC 6.0 anyway...
*/
template <int num_dimensions, typename T>
inline BasicCoordinate<num_dimensions, int>
get_min_indices(const Array<num_dimensions, T>& a)
{
  return detail::get_min_indices_help(detail::test_if_1d<num_dimensions>(), a);
}

template <int num_dimensions, typename T, int num_dimensions2>
inline bool
next(BasicCoordinate<num_dimensions, int>& index, const Array<num_dimensions2, T>& a)
{
  return detail::next_help(detail::test_if_1d<num_dimensions>(), index, a);
}

template <int num_dimensions, int num_dimensions2, typename elemT>
inline const Array<num_dimensions - num_dimensions2, elemT>&
get(const Array<num_dimensions, elemT>& a, const BasicCoordinate<num_dimensions2, int>& c)
{
  return get(a[c[1]], cut_first_dimension(c));
}

template <int num_dimensions, typename elemT>
inline const elemT&
get(const Array<num_dimensions, elemT>& a, const BasicCoordinate<num_dimensions, int>& c)
{
  return a[c];
}
template <int num_dimensions, typename elemT>
inline const Array<num_dimensions - 1, elemT>&
get(const Array<num_dimensions, elemT>& a, const BasicCoordinate<1, int>& c)
{
  return a[c[1]];
}

END_NAMESPACE_STIR
