//
// $Id$: $Date$
//
#ifndef __interpolate_h__
#define __interpolate_h__


#include "VectorWithOffset.h"

/*
  This declares an implementation of 'overlap' interpolation on 
  arbitrary data types (using templates).
  This type of interpolation considers the data as the samples of
  a step-wise function. The interpolated array again represents a
  step-wise function, such that the counts (i.e. integrals) are
  preserved.

  The spacing between the new points is determined by the 'zoom'
  parameter: e.g. zoom <1 stretches the bin size with a factor 1/zoom.

  'offset' (measured in 'units' of the in_data) allows to shift the
  range of values you want to compute. In particular, having
	offset > 0
  shifts the data to the left (if in_data and out_data have the same
  range of indices).
  Note that a similar (but less general) effect to using 'offset' can be
  achieved by adjusting the min and max indices of the out_data.
  For an index x_out (in out_data coordinates), the corresponding
  in_data coordinates is x_in = x_out/zoom  + offset
  (The convention is used that the 'bins' are centered around the
  coordinate value.)

  'assign_rest_with_zeroes'
  If 'false' does not set values in out_data which do not overlap with
  in_data.
  If 'true' those data are set to 0. (The effect being the same as first
  doing out_data.fill(0) before calling overlap_interpolate).

  Warning: when T involves integral types, there is no rounding 
  (but truncation)
  (TODO ?)

  Examples:
  in_data = {a,b,c,d} indices from 0 to 3
  zoom = .5
  offset = .5
  out_data = {a+b, c+d} indices from 0 to 1

  in_data = {a,b,c,d} indices from 0 to 3
  zoom = .5
  offset = -.5
  out_data = {a,b+c,d} indices from 0 to 2

  in_data = {a,b,c} indices from -1 to 1
  zoom = .5
  offset = 0
  out_data = {a/2, a/2+b+c/2, c/2} indices from -1 to 1

*/

template <typename T>
void
overlap_interpolate(VectorWithOffset<T>& out_data, 
		    const VectorWithOffset<T>& in_data,
		    const float zoom, 
		    const float offset, 
		    const bool assign_rest_with_zeroes = true);
#endif
