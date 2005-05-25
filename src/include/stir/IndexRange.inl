//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \brief inline definitions for the IndexRange class

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

*/
#include <algorithm>

START_NAMESPACE_STIR

/***************************************
 n-D version
 ***************************************/

template <int num_dimensions>
IndexRange<num_dimensions>::IndexRange()
  : base_type(),
    is_regular_range(regular_true)
  {}

template <int num_dimensions>
IndexRange<num_dimensions>::IndexRange(const IndexRange<num_dimensions>& range)
  : base_type(range),
    is_regular_range(range.is_regular_range)
  {}

template <int num_dimensions>
IndexRange<num_dimensions>::IndexRange(const base_type& range)
  : base_type(range),
    is_regular_range(regular_to_do)
  {}

template <int num_dimensions>
IndexRange<num_dimensions>::
IndexRange(
	   const BasicCoordinate<num_dimensions, int>& min_v,
	   const BasicCoordinate<num_dimensions, int>& max_v)
  : base_type(min_v[1], max_v[1]),
    is_regular_range(regular_true)
{
  const IndexRange<num_dimensions-1>
    lower_dims(cut_first_dimension(min_v), cut_first_dimension(max_v));
  this->fill(lower_dims);
}

template <int num_dimensions>
IndexRange<num_dimensions>::
IndexRange(const BasicCoordinate<num_dimensions, int>& sizes)
  : base_type(sizes[1]),
    is_regular_range(regular_true)
{
  const IndexRange<num_dimensions-1>
    lower_dims(cut_first_dimension(sizes));
  this->fill(lower_dims);
}

template <int num_dimensions>
bool
IndexRange<num_dimensions>::
  operator==(const IndexRange<num_dimensions>& range2) const
{
  return 
    this->get_min_index() == range2.get_min_index() &&
    this->get_length() == range2.get_length() &&
#ifndef STIR_NO_NAMESPACES  
    // "using std::equal" didn't work for VC 6...
    std::equal(this->begin(), this->end(), range2.begin());
#else
    equal(this->begin(), this->end(), range2.begin());
#endif
}


template <int num_dimensions>
bool
IndexRange<num_dimensions>::
  operator!=(const IndexRange<num_dimensions>& range2) const
{
  return !(*this==range2);
}

template <int num_dimensions>
bool
IndexRange<num_dimensions>::
  is_regular() const
{
  switch (is_regular_range)
  {
    case regular_true: 
       return true;       
    case regular_false:
       return false;
    case regular_to_do:
       {
	 BasicCoordinate<num_dimensions, int> min;
	 BasicCoordinate<num_dimensions, int> max;
	 return
	   get_regular_range(min,max);
       }
  }
  // although we never get here, VC insists on a return value...
  // we check anyway
  assert(false);
  return true;
}

#ifdef STIR_NO_MUTABLE

template <int num_dimensions>
bool
IndexRange<num_dimensions>::
  is_regular() 
{
  switch (is_regular_range)
  {
    case regular_true: 
       return true;       
    case regular_false:
       return false;
    case regular_to_do:
       {
	 BasicCoordinate<num_dimensions, int> min;
	 BasicCoordinate<num_dimensions, int> max;
	 return
	   get_regular_range(min,max);
       }
  }
  // although we never get here, VC insists on a return value...
  // we check anyway
  assert(false);
  return true;
}
#endif // STIR_NO_MUTABLE
/***************************************
 1D version
 ***************************************/

IndexRange<1>::IndexRange()
: min(0), max(0)
{}

IndexRange<1>::IndexRange(const int min_v, const int max_v)
: min(min_v), max(max_v)
{}

IndexRange<1>::IndexRange(const BasicCoordinate<1,int>& min_v, 
			  const BasicCoordinate<1,int>& max_v)
: min(min_v[1]), max(max_v[1])
{}

IndexRange<1>::IndexRange(const int length)
: min(0), max(length-1)
{}

IndexRange<1>::IndexRange(const BasicCoordinate<1,int>& size)
: min(0), max(size[1]-1)
{}

int IndexRange<1>::get_min_index() const
{ return min;}

int 
IndexRange<1>::get_max_index() const
{ return max;}

int 
IndexRange<1>::get_length() const
{ return max-min+1; }

bool
IndexRange<1>::operator==(const IndexRange<1>& range2) const
{
  return 
    get_min_index() == range2.get_min_index() &&
    get_length() == range2.get_length();
}

bool
IndexRange<1>::is_regular() const
{
  // 1D case: always true
  return true;
}

bool 
IndexRange<1>::get_regular_range(
    BasicCoordinate<1, int>& min_v,
    BasicCoordinate<1, int>& max_v) const
{
  // somewhat complicated as we can't assign ints to BasicCoordinate<1,int>
  BasicCoordinate<1, int> tmp;
  tmp[1] = min;
  min_v = tmp;
  tmp[1] = max;
  max_v = tmp;
  return true;
}

void 
IndexRange<1>::
resize(const int min_index, const int max_index)
{
  min = min_index;
  max = max_index;
}
END_NAMESPACE_STIR
