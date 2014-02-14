//
//
/*!
  \file 
  \ingroup Array 
  \brief implementations for the stir::IndexRange class

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
#include "stir/IndexRange.h"
#include <algorithm>

START_NAMESPACE_STIR

#ifndef STIR_NO_MUTABLE


template <int num_dimensions>
bool
IndexRange<num_dimensions>::get_regular_range(
     BasicCoordinate<num_dimensions, int>& min,
     BasicCoordinate<num_dimensions, int>& max)  const
{
  // check if empty range    
  if (base_type::begin() == base_type::end())
  {
#ifndef STIR_NO_NAMESPACES  
    std::fill(min.begin(), min.end(), 0);
    std::fill(max.begin(), max.end(),-1);
#else
    // gcc 2.8.1 needs ::fill, otherwise it gets confused with VectorWithOffset::fill
    ::fill(min.begin(), min.end(), 0);
    ::fill(max.begin(), max.end(),-1);
#endif
    return true;
  }

  // if not a regular range, exit
  if (is_regular_range == regular_false)
    return false;

  typename base_type::const_iterator iter=base_type::begin();

  BasicCoordinate<num_dimensions-1, int> lower_dim_min;
  BasicCoordinate<num_dimensions-1, int> lower_dim_max;
  if (!iter->get_regular_range(lower_dim_min, lower_dim_max))
    return false;

  if (is_regular_range == regular_to_do)
  {
    // check if all lower dimensional ranges have same regular range  
    BasicCoordinate<num_dimensions-1, int> lower_dim_min_try;
    BasicCoordinate<num_dimensions-1, int> lower_dim_max_try;
    
    for (++iter; iter != base_type::end(); ++iter)
    {
      if (!iter->get_regular_range(lower_dim_min_try, lower_dim_max_try))
      {
	is_regular_range = regular_false;
	return false;
      }
#ifndef STIR_NO_NAMESPACES
      if (!std::equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin()) ||
	  !std::equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
#else
	if (!equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin()) ||
            !equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
#endif
      {
	is_regular_range = regular_false;
	return false;
      }
    }
    // yes, they do
    is_regular_range = regular_true;
  }

#if defined(_MSC_VER) && _MSC_VER<1200
  // bug in VC++ 5.0, needs explicit template args
  min = join<num_dimensions-1,int>(base_type::get_min_index(), lower_dim_min);
  max = join<num_dimensions-1,int>(base_type::get_max_index(), lower_dim_max);
#elif defined( __GNUC__) && (__GNUC__ == 2 && __GNUC_MINOR__ < 9)
  // work around gcc 2.8.1 bug. 
  // It cannot call 'join' (it generates a bad mangled name for the function)
  // So, we explicitly insert the code here
  *min.begin() = base_type::get_min_index();
   copy(lower_dim_min.begin(), lower_dim_min.end(), min.begin()+1);
  *max.begin() = base_type::get_max_index();
   copy(lower_dim_max.begin(), lower_dim_max.end(), max.begin()+1);
#else
   // lines for good compilers...
  min = join(base_type::get_min_index(), lower_dim_min);
  max = join(base_type::get_max_index(), lower_dim_max);  
#endif
  return true;
}

#else // STIR_NO_MUTABLE

template <int num_dimensions>
bool
IndexRange<num_dimensions>::get_regular_range(
     BasicCoordinate<num_dimensions, int>& min,
     BasicCoordinate<num_dimensions, int>& max)  const
{
  // check if empty range    
  if (base_type::begin() == base_type::end())
  {
#ifndef STIR_NO_NAMESPACES  
    std::fill(min.begin(), min.end(), 0);
    std::fill(max.begin(), max.end(),-1);
#else
    fill(min,base_type::begin(), min.end(), 0);
    fill(max,base_type::begin(), max.end(),-1);
#endif
    return true;
  }

  // if not a regular range, exit
  if (is_regular_range == regular_false)
    return false;

  base_type::const_iterator iter=base_type::begin();

  BasicCoordinate<num_dimensions-1, int> lower_dim_min;
  BasicCoordinate<num_dimensions-1, int> lower_dim_max;
  if (!iter->get_regular_range(lower_dim_min, lower_dim_max))
    return false;

  if (is_regular_range == regular_to_do)
  {
    // check if all lower dimensional ranges have same regular range  
    BasicCoordinate<num_dimensions-1, int> lower_dim_min_try;
    BasicCoordinate<num_dimensions-1, int> lower_dim_max_try;
    
    for (++iter; iter != base_type::end(); ++iter)
    {
      if (!iter->get_regular_range(lower_dim_min_try, lower_dim_max_try))
      {
	//is_regular_range = regular_false;
	return false;
      }
#ifndef STIR_NO_NAMESPACES
      if (!std::equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin()) ||
	  !std::equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
#else
	if (!equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin()) ||
            !equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
#endif
      {
	//is_regular_range = regular_false;
	return false;
      }
    }
    // yes, they do
    //is_regular_range = regular_true;
  }

#if defined(_MSC_VER) && _MSC_VER<1200
  // bug in VC++ 5.0, needs explicit template args
  min = join<num_dimensions-1,int>(base_type::get_min_index(), lower_dim_min);
  max = join<num_dimensions-1,int>(base_type::get_max_index(), lower_dim_max);
#else
  min = join(base_type::get_min_index(), lower_dim_min);
  max = join(base_type::get_max_index(), lower_dim_max);
#endif
  return true;
}

template <int num_dimensions>
bool
IndexRange<num_dimensions>::get_regular_range(
     BasicCoordinate<num_dimensions, int>& min,
     BasicCoordinate<num_dimensions, int>& max)
{
  // check if empty range    
  if (base_type::begin() == base_type::end())
  {
#ifndef STIR_NO_NAMESPACES  
    std::fill(min.begin(), min.end(), 0);
    std::fill(max.begin(), max.end(),-1);
#else
    fill(min.begin(), min.end(), 0);
    fill(max.begin(), max.end(),-1);
#endif
    return true;
  }

  // if not a regular range, exit
  if (is_regular_range == regular_false)
    return false;

  base_type::iterator iter=base_type::begin();

  BasicCoordinate<num_dimensions-1, int> lower_dim_min;
  BasicCoordinate<num_dimensions-1, int> lower_dim_max;
  if (!iter->get_regular_range(lower_dim_min, lower_dim_max))
    return false;

  if (is_regular_range == regular_to_do)
  {
    // check if all lower dimensional ranges have same regular range  
    BasicCoordinate<num_dimensions-1, int> lower_dim_min_try;
    BasicCoordinate<num_dimensions-1, int> lower_dim_max_try;
    
    for (++iter; iter != base_type::end(); ++iter)
    {
      if (!iter->get_regular_range(lower_dim_min_try, lower_dim_max_try))
      {
	is_regular_range = regular_false;
	return false;
      }
#ifndef STIR_NO_NAMESPACES
      if (!std::equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin()) ||
	  !std::equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
#else
	if (!equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin()) ||
            !equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
#endif
      {
	is_regular_range = regular_false;
	return false;
      }
    }
    // yes, they do
    is_regular_range = regular_true;
  }

#if defined(_MSC_VER) && _MSC_VER<1200
  // bug in VC++ 5.0, needs explicit template args
  min = join<num_dimensions-1,int>(base_type::get_min_index(), lower_dim_min);
  max = join<num_dimensions-1,int>(base_type::get_max_index(), lower_dim_max);
#else
  min = join(base_type::get_min_index(), lower_dim_min);
  max = join(base_type::get_max_index(), lower_dim_max);
#endif
  return true;
}

#endif // STIR_NO_MUTABLE


/***************************************************
 instantiations
 ***************************************************/

template class IndexRange<2>;
template class IndexRange<3>;
template class IndexRange<4>;
END_NAMESPACE_STIR
