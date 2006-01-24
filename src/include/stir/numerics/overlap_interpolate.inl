//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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

  \brief Implementation of inline versions of stir::overlap_interpolate
  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include <algorithm>
#include "boost/iterator/iterator_traits.hpp"

START_NAMESPACE_STIR
template <typename out_iter_t, typename out_coord_iter_t,
	  typename in_iter_t, typename in_coord_iter_t>
void
 overlap_interpolate(const out_iter_t out_begin, const out_iter_t out_end, 
		     const out_coord_iter_t out_coord_begin, const out_coord_iter_t out_coord_end,
		     const in_iter_t in_begin, in_iter_t in_end,
		     const in_coord_iter_t in_coord_begin, const in_coord_iter_t in_coord_end,
		     const bool only_add_to_output,
		     const bool assign_rest_with_zeroes)
{

  if (out_begin == out_end)
    return;
  if (in_begin == in_end)
    return;

  // check sizes.
  // these asserts can be disabled if we ever use this function with iterators
  // that don't support the next 2 lines
  assert(out_coord_end - out_coord_begin - 1 == (out_end - out_begin)); 
  assert(in_coord_end - in_coord_begin - 1 == (in_end - in_begin)); 

  out_iter_t out_iter = out_begin;
  out_coord_iter_t out_coord_iter = out_coord_begin;

  in_iter_t in_iter = in_begin;
  in_coord_iter_t in_coord_iter = in_coord_begin;

  // skip input to the left of the output range
  assert(in_coord_iter+1 != in_coord_end);
  while (*(in_coord_iter+1) <= *out_coord_iter)
    {
      ++in_coord_iter; ++in_iter; 
      if (in_coord_iter+1 == in_coord_end)
	return;
    }

  // skip output to the left of the input range
  assert(out_coord_iter+1 != out_coord_end);
  while (*(out_coord_iter+1) <= *in_coord_iter)
    {
      if (!only_add_to_output && assign_rest_with_zeroes)
	*out_iter *= 0; // note: use *= such that it works for multi-dim arrays
      ++out_coord_iter; ++out_iter; 
      if (out_coord_iter+1 == out_coord_end)
	return;
    }

  // now first in-box overlaps guaranteed with first out-box
  assert(*(out_coord_iter+1) > *in_coord_iter);
  assert(*(in_coord_iter+1) > *out_coord_iter);

  // a typedef for the coordinate type
  typedef typename boost::iterator_value<out_coord_iter_t>::type coord_t;

  // find small number for comparisons.
  // we'll take it 1000 times smaller than the minimum of the average out_box size or in_box size
  const coord_t epsilon = 
    std::min(((*(out_coord_end-1)) - (*out_coord_begin)) /
	     ((out_coord_end-1 - out_coord_begin)*1000),
	     ((*(in_coord_end-1)) - (*in_coord_begin)) /
	     ((in_coord_end-1 - in_coord_begin)*1000));
  
  // do actual interpolation
  // we walk through the boxes, checking the overlap.
  // after each step, we'll advance either in_iter or out_iter.
  coord_t current_coord = std::max(*in_coord_iter, *out_coord_iter);
  bool first_time_for_this_out_box = true;
  while (true)
    {
      // right edge of in-box is beyond out-box
      const bool in_beyond_out =  
	*(in_coord_iter+1) > *(out_coord_iter+1);
      const coord_t new_coord =
	in_beyond_out ? *(out_coord_iter+1) : *(in_coord_iter+1);
#ifndef STIR_OVERLAP_NORMALISATION
      const coord_t overlap = (new_coord - current_coord);
#else
      const coord_t overlap = (new_coord - current_coord)/
      	(*(out_coord_iter+1) - *(out_coord_iter));
#endif
      assert(overlap>-epsilon);

      if (!only_add_to_output && first_time_for_this_out_box)
	{
	  if (overlap>epsilon)
	    *out_iter = *in_iter * overlap;
	  else
	    *out_iter *= 0;
	  first_time_for_this_out_box = false;
	}
      else
	{
	  if (overlap>epsilon)
	    *out_iter += *in_iter * overlap;
	}
      current_coord = new_coord;
      if (in_beyond_out)
	{
	  ++out_coord_iter; ++out_iter; 
	  if (out_iter == out_end)
	    {
	      assert(out_coord_iter+1 == out_coord_end);	      
	      return; // all out-boxes are done
	    }
	  first_time_for_this_out_box = true;
	}
      else
	{
	  ++in_coord_iter; ++in_iter; 
	  if (in_iter == in_end)
	    break;
	}
    } // end of while

  assert(in_coord_iter+1 == in_coord_end);	      
  // fill rest of output with 0
  if (!only_add_to_output && assign_rest_with_zeroes)
    {
      while (true)
	{
	  ++out_iter; 
#ifndef NDEBUG
	  // increment just so we can check at the end that there is 
	  // one more out_coord_iter than out_iter
	  ++out_coord_iter; 
#endif
	  if (out_iter == out_end)
	    break;
	  *out_iter *= 0;  // note: use *= such that it works for multi-dim arrays
	}
      assert(out_coord_iter+1 == out_coord_end);	      
    }
}

END_NAMESPACE_STIR
