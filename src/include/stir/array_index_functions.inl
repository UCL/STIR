//
// $Id$
//
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
/*!
  \file 
  \ingroup Array
 
  \brief implementation of functions in stir/array_index_functions.h

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
inline
BasicCoordinate<num_dimensions, int>
get_min_indices_help(is_not_1d, const Array<num_dimensions, T>& a)
{
  if (a.get_min_index()<=a.get_max_index())
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
inline
BasicCoordinate<1, int>
get_min_indices_help(is_1d, const Array<1, T>& a)
{
  BasicCoordinate<1, int> result;
  result[1] = a.get_min_index();
  return result;
}

template <int num_dimensions2, typename T>
inline
bool 
next_help(is_1d, BasicCoordinate<1, int>& index, 
     const Array<num_dimensions2, T>& a)
{
  if (a.get_min_index()>a.get_max_index())
    return false;
  assert(index[1]>=a.get_min_index());
  assert(index[1]<=a.get_max_index());
  index[1]++;
  return index[1]<=a.get_max_index();
}

template <typename T, int num_dimensions, int num_dimensions2>
inline
bool 
next_help(is_not_1d,
          BasicCoordinate<num_dimensions, int>& index, 
          const Array<num_dimensions2, T>& a)
{
  if (a.get_min_index()>a.get_max_index())
    return false;
  BasicCoordinate<num_dimensions-1, int> upper_index= cut_last_dimension(index);
  assert(index[num_dimensions]>=get(a,upper_index).get_min_index());
  assert(index[num_dimensions]<=get(a,upper_index).get_max_index());
  index[num_dimensions]++;
  if (index[num_dimensions]<=get(a,upper_index).get_max_index())
    return true;
  if (!next(upper_index, a))
    return false;
  index=join(upper_index, get(a,upper_index).get_min_index());
  return true;
}

} // end of namespace detail


/* Now define the functions in the stir namespace in terms of the above.
   Also define get() for which I didn't bother to try the work-arounds,
   as they don't work for VC 6.0 anyway...
*/
template <int num_dimensions, typename T>
inline
BasicCoordinate<num_dimensions, int>
get_min_indices(const Array<num_dimensions, T>& a)
{
  return detail::get_min_indices_help(detail::test_if_1d<num_dimensions>(), a);
}
#if !defined(_MSC_VER) || _MSC_VER>1200

template <int num_dimensions, typename T, int num_dimensions2>
inline
bool 
next(BasicCoordinate<num_dimensions, int>& index, 
     const Array<num_dimensions2, T>& a)
{
  return detail::next_help(detail::test_if_1d<num_dimensions>(), index, a);
}

template <int num_dimensions, int num_dimensions2, typename elemT>
inline
const Array<num_dimensions-num_dimensions2,elemT>&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions2,int> &c) 
{
  return get(a[c[1]],cut_first_dimension(c)); 
}

template <int num_dimensions, typename elemT>
inline
const elemT&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions,int> &c) 
{
  return a[c];
}			
template <int num_dimensions, typename elemT>
inline
const Array<num_dimensions-1,elemT>&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<1,int> &c) 
{
  return a[c[1]]; 
}			


#else

/* terrible code for VC 6.0
   Stop reading now unless you have that compiler (or something else severely broken).
   This code works right now, (September 2004) but will no longer be supported in later versions.

   We need to list all dimensions explicitly. Sigh.
   I do this with macros to avoid too much repetition.

   Aside from that, the actual code really should be the same.
*/

#define GET(num_dimensions, num_dimensions2) \
template <typename elemT> \
inline \
const Array<num_dimensions-num_dimensions2,elemT>& \
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions2,int> &c)  \
{ \
  return get(a[c[1]],cut_first_dimension(c)); \
}

#define GET_EQUAL_DIM(num_dimensions) \
template <typename elemT> \
inline \
const elemT& \
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions,int> &c) \
{ \
  return a[c]; \
}		

#define GET_1D(num_dimensions) \
template <typename elemT> \
inline \
const Array<num_dimensions-1,elemT>& \
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<1,int> &c) \
{ \
  return a[c[1]]; \
}			

GET_1D(2)
GET_1D(3)
GET_EQUAL_DIM(1)
GET_EQUAL_DIM(2)
GET_EQUAL_DIM(3)
GET(3,2)

#undef GET_1D
#undef GET_EQUAL_DIM
#undef GET


#define DEFINE_NEXT_1D(num_dimensions2) \
template <typename T> \
inline bool \
next(BasicCoordinate<1, int>& index, \
     const Array<num_dimensions2, T>& a) \
{ if (a.get_min_index()>a.get_max_index())\
    return false;\
  assert(index[1]>=a.get_min_index()); \
  assert(index[1]<=a.get_max_index()); \
  index[1]++;\
  return index[1]<=a.get_max_index();\
}

#define DEFINE_NEXT(num_dimensions, num_dimensions2) \
template <typename T> \
inline bool \
next(BasicCoordinate<num_dimensions, int>& index, \
     const Array<num_dimensions2, T>& a) \
{\
  if (a.get_min_index()>a.get_max_index()) \
    return false; \
  BasicCoordinate<num_dimensions-1, int> upper_index= cut_last_dimension(index);\
  assert(index[num_dimensions]>=get(a,upper_index).get_min_index());\
  assert(index[num_dimensions]<=get(a,upper_index).get_max_index());\
  index[num_dimensions]++;\
  if (index[num_dimensions]<=get(a,upper_index).get_max_index())\
    return true;\
  if (!next(upper_index, a))\
    return false;\
  index=join(upper_index, get(a,upper_index).get_min_index());\
}



DEFINE_NEXT_1D(1)
DEFINE_NEXT_1D(2)
DEFINE_NEXT_1D(3)
DEFINE_NEXT(2,2)
DEFINE_NEXT(2,3)
DEFINE_NEXT(3,3)

#undef DEFINE_NEXT_1D
#undef DEFINE_NEXT

#endif // end of VC 6.0 code

END_NAMESPACE_STIR
