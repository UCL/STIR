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

  \brief Implementation of inline versions of overlap_interpolate()
  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include <algorithm>

START_NAMESPACE_STIR
template <typename out_iter_t, typename out_coord_iter_t,
	  typename in_iter_t, typename in_coord_iter_t>
void
 overlap_interpolate(const out_iter_t out_begin, const out_iter_t out_end, 
		    const out_coord_iter_t out_coord_begin, const out_coord_iter_t out_coord_end,
		    const in_iter_t in_begin, in_iter_t in_end,
		    const in_coord_iter_t in_coord_begin, const in_coord_iter_t in_coord_end,
		    const bool assign_rest_with_zeroes)
{

  if (out_begin == out_end)
    return;
  if (in_begin == in_end)
    return;

  out_iter_t out_iter = out_begin;
  out_coord_iter_t out_coord_iter = out_coord_begin;

  in_iter_t in_iter = in_begin;
  in_coord_iter_t in_coord_iter = in_coord_begin;

  // skip input to the left of the output range
  assert(in_coord_iter+1 != in_coord_end);
  while (true)
    {
      if (*(in_coord_iter+1) > *out_coord_iter)
	break;
      ++in_coord_iter; ++in_iter; 
      if (in_coord_iter+1 == in_coord_end)
	return;
    }

  // skip output to the left of the input range
  assert(out_coord_iter+1 != out_coord_end);
  while (true)
    {
      if (*(out_coord_iter+1) > *in_coord_iter)
	break;
      if (assign_rest_with_zeroes)
	*out_iter *= 0; // note: use *= such that it works for multi-dim arrays
      ++out_coord_iter; ++out_iter; 
      if (out_coord_iter+1 == out_coord_end)
	return;
    }

  // now first in-box overlaps guaranteed with first out-box
  assert(*(out_coord_iter+1) > *in_coord_iter);
  assert(*(in_coord_iter+1) > *out_coord_iter);

  // do actual interpolation
  // we walk through the boxes, checking the overlap.
  // after each step, we'll advance either in_iter or out_iter.
  float current_coord = std::max(*in_coord_iter, *out_coord_iter);
  bool first_time_for_this_out_box = true;
  while (true)
    {
      if (*(in_coord_iter+1) > *(out_coord_iter+1))
	{
	  // right edge of in-box is beyond out-box
	  const float overlap =  *(out_coord_iter+1) - current_coord;
	  if (first_time_for_this_out_box)
	    {
	      if (overlap>0)
		 *out_iter = *in_iter * overlap;
	       else
		 *out_iter *= 0;
	    }
	  else
	    {
	      if (overlap>0)
		*out_iter += *in_iter * overlap;
	    }
	  current_coord = *(out_coord_iter+1);
	  ++out_coord_iter; ++out_iter; 
	  if (out_iter == out_end)
	    {
	      assert(out_coord_iter+1 == out_coord_end);	      
	      return; // all out-boxes are done
	    }
	  assert (out_coord_iter+1 != out_coord_end);
	  first_time_for_this_out_box = true;
	}
      else
	{
	  // right edge of in-box is inside out-box
	  const float overlap =  *(in_coord_iter+1) - current_coord;
	  if (first_time_for_this_out_box)
	    {
	      first_time_for_this_out_box = false;
	       if (overlap>0)
		 *out_iter = *in_iter * overlap;
	       else
		 *out_iter *= 0;
	    }
	  else
	    {
	      if (overlap>0)
		*out_iter += *in_iter * overlap;
	    }
	  current_coord = *(in_coord_iter+1);
	  ++in_coord_iter; ++in_iter; 
	  if (in_iter == in_end)
	    break;
	  assert (in_coord_iter+1 != in_coord_end);
	}
    } // end of while

  assert(in_coord_iter+1 == in_coord_end);	      
  // fill rest of output with 0
  if (assign_rest_with_zeroes)
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
    }
  assert(out_coord_iter+1 == out_coord_end);	      
}

END_NAMESPACE_STIR
