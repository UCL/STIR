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
  \ingroup buildblock

  \brief Implementation of overlap_interpolate

  \author Kris Thielemans (with suggestions by Alexey Zverovich)
  \author PARAPET project

  $Date$
  $Revision$
*/
#include "stir/interpolate.h"

START_NAMESPACE_STIR

/*! 

  This is an implementation of 'overlap' interpolation on 
  arbitrary data types (using templates).

  This type of interpolation considers the data as the samples of
  a step-wise function. The interpolated array again represents a
  step-wise function, such that the counts (i.e. integrals) are
  preserved.

  \param zoom
  The spacing between the new points is determined by the 'zoom'
  parameter: e.g. zoom less than 1 stretches the bin size with a factor 1/zoom.

  \param offset (measured in 'units' of the in_data) allows to shift the
  range of values you want to compute. In particular, having positive
  offset shifts the data to the left (if in_data and out_data have the same
  range of indices).
  Note that a similar (but less general) effect to using 'offset' can be
  achieved by adjusting the min and max indices of the out_data.
  
  \param assign_rest_with_zeroes
  If \c false does not set values in \c out_data which do not overlap with
  \c in_data.
  If \c true those data are set to 0. (The effect being the same as first
  doing \c out_data.fill(0) before calling overlap_interpolate).

  For an index x_out (in \c out_data coordinates), the corresponding
  \c in_data coordinates is <code>x_in = x_out/zoom  + offset</code>
  (The convention is used that the 'bins' are centered around the
  coordinate value.)

  \warning when T involves integral types, there is no rounding 
  but truncation.

  \par Examples:

  \verbatim
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
  \endverbatim
  
  \par Implementation details:

  Because this implementation works for arbitrary (numeric) types T, it
  is slightly more complicated than would be necessary for (say) floats.
  In particular,<br> 
  - we do our best to avoid creating temporary objects of type T<br>
  - we zero values by using multiplication with 0 <br>
  (actually we use T::operator*=(0)). This is to allow the case where
  T::operator=(int) does not exist (in particular, in our higher 
  dimensional arrays).


  \par History:
  <ul>
  <li> first version by Kris Thielemans with suggestions by Alexey Zverovich.
    (loosely based on a 1D version by Claire Labbe)
  </ul>
*/

template <typename T>
void
overlap_interpolate(VectorWithOffset<T>& out_data, 
		    const VectorWithOffset<T>& in_data,
		    const float zoom, 
		    const float offset, 
		    const bool assign_rest_with_zeroes)
{ 
  assert(zoom>0);
  
  // First check trivial case
  if (zoom==1.F && offset==0.F && 
    in_data.get_min_index()==out_data.get_min_index() &&
    in_data.get_max_index()==out_data.get_max_index())
  {
    out_data = in_data;
    return;
  }
  
  if(zoom>=1)
  {
    // Shrinking to a smaller bin size    
    
    // start at the first 'in' bin that overlaps the 'out' data.
    // compute by checking by comparing its position with
    // the position of the left edge of the first 'out' bin:
    // left_edge = (out_data.get_min_index() - .5)/zoom + offset
    // x1 -.5 <= left_edge < x1+.5
    int x2 = out_data.get_min_index();
    int x1 = (int)floor((x2 - .5)/zoom + offset + .5);
    
    // the next variable holds the difference between the coordinates
    // of the right edges of the 'in' bin and the 'out' bin, computed
    // in a coordinate system where the 'out' voxels are unit distance apart
    double diff_between_right_edges = 
      zoom*(x1-offset+.5) - (x2 + .5);
    
    for(; 
        x2 <= out_data.get_max_index(); 
        x2++, diff_between_right_edges--)
    {
      
      if(x1> in_data.get_max_index())
      {
	// just fill out_data with 0, 
	// no need to check/update diff_between_right_edges 
	if (assign_rest_with_zeroes) 
	  out_data[x2] *= 0;
	
	continue;
      }
      
      assert(diff_between_right_edges<= zoom/*+epsilon*/);
      assert(diff_between_right_edges>= -1/*epsilon*/);
      
      if (diff_between_right_edges >= 0)
      {
	if(x1 >= in_data.get_min_index()) 
	{
	  out_data[x2] = in_data[x1];
	  out_data[x2] /= zoom;
	}
	else 
	{
	  if (assign_rest_with_zeroes) 
	    out_data[x2] *= 0;
	}
      }
      else
      {	
        /*	
          Set out_data[x2] according to
      
          T V1; // bin value at x1
          T V2; // bin value at x1+1	
          out_data[x2] = (V1+diff_between_right_edges*(V1-V2))/zoom;
       
          The lines below are more complicated because
	  - testing if x1, x1+1 are inside the range of in_data
	  - everything is done without creating temporary objects of type T
        */
	if(x1 >= in_data.get_min_index()) 
	{
	  out_data[x2] = in_data[x1]; 
	  out_data[x2] *= static_cast<float>(1/diff_between_right_edges + 1); // note conversion to float to avoid compiler warnings in case that T is float (or a float array)
	}
	else 
	{
	  if (assign_rest_with_zeroes) 	
	    out_data[x2] *= 0;	  
	}	
	if(x1+1 <= in_data.get_max_index() && x1+1>=in_data.get_min_index()) 
	{
	  out_data[x2] -= in_data[x1+1];
	}
	
	out_data[x2] *= static_cast<float>(diff_between_right_edges/zoom);
	
	// advance 'in' bin
	x1++;
	diff_between_right_edges += zoom;
      }
    }// End of for x2
    
  }
  else
  { 
    // case zoom <1 : stretching the bin size     
    // start 1 before the first 'in' bin that overlaps the 'out' data.
    // compute by comparing its position with
    // the position of the left edge of the first 'out' bin:
    // left_edge = (out_data.get_min_index() - .5)/zoom + offset
    // x1-.5 <= left_edge < x1+.5
    
    const float inverse_zoom = 1/zoom;
    
    // current coordinate in out_data
    int x2 = out_data.get_min_index();
    // current coordinate in in_data
    int x1 = (int)floor((x2 - .5)*inverse_zoom  + offset + .5);
    
    // the next variable holds the difference between the coordinates
    // of the right edges of the 'in' bin and the 'out' bin, computed
    // in a coordinate system where the 'in' bins are unit distance apart
    double diff_between_right_edges = (x1-offset+.5) - (x2 + .5)*inverse_zoom;
    
    // we will loop over x1 to update out_data[x2] from in_data[x1]
    // however, we first check if the left edge of the first in_data is
    // to the left of the left edge of the first out_data.
    // If so, will first subtract the contribution of the 'in' bin that
    // lies outside the 'out' bin. In the loop later, this part of the
    // 'in' bin will be added again.
    {
      const double diff_between_left_edges =
	diff_between_right_edges-1+inverse_zoom;
      if (diff_between_left_edges < 0 &&
	x1 >= in_data.get_min_index() && x1 <= in_data.get_max_index() )
      {
	out_data[x2] = in_data[x1];
	out_data[x2] *= static_cast<float>(diff_between_left_edges);
      }
      else
      {
	if (assign_rest_with_zeroes) 	
	  out_data[x2] *= 0;	  
      }
    }

    for (;
         x1 <= in_data.get_max_index(); 
         x1++, diff_between_right_edges++)
    {
      assert(diff_between_right_edges<= 1/*+epsilon*/);
      assert(diff_between_right_edges>= -inverse_zoom/*-epsilon*/);
      
      if (diff_between_right_edges <= 0)
      {
	// 'in' bin fits entirely in the 'out' bin
	if (x1 >= in_data.get_min_index())
	  out_data[x2] += in_data[x1];
      }
      else 
      {
	// dx = fraction of 'in' bin that lies within 'out' bin
	const double dx= 1- diff_between_right_edges;
	// update current 'out' bin
	if (x1 >= in_data.get_min_index())
	{
	  // out_data[x2] += in_data[x1]*dx;
	  if (fabs(dx) > 1e-5)
	  {
	    out_data[x2] /= static_cast<float>(dx);
	    out_data[x2] += in_data[x1];
	    out_data[x2] *= static_cast<float>(dx);
	  }
	}
	// next bin
	x2++;
	diff_between_right_edges -= inverse_zoom;
	// update this one with the rest
	if (x2<= out_data.get_max_index())
	{
	  if (x1 >= in_data.get_min_index())
	  {
	    // out_data[x2] = in_data[x1]*(1-dx);
	    out_data[x2] = in_data[x1];
	    out_data[x2] *= static_cast<float>((1-dx));
	  }
	  else if (assign_rest_with_zeroes) 
	  {
	    out_data[x2] *= 0;
	  }
	}
	else
	{
	  // x2 goes out of range, so we can just as well stop
	  break;
	}
      }
      
    }// End of for x1
    
    if (assign_rest_with_zeroes) 
    {
      // set rest of out_data to 0
      for (x2++;
           x2 <= out_data.get_max_index(); 
           x2++)
	out_data[x2] *= 0;
    }
    
    
  }// End of if(zoom>1)
  
}



/************************************************
 Instantiations
 ************************************************/

template
void 
overlap_interpolate<>(VectorWithOffset<float>& out_data, 
		      const VectorWithOffset<float>& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

template
void 
overlap_interpolate<>(VectorWithOffset<double>& out_data, 
		      const VectorWithOffset<double>& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);
END_NAMESPACE_STIR

 // TODO remove
#if defined(OLDDESIGN)

#include "stir/Tensor2D.h"

template
void 
overlap_interpolate<>(VectorWithOffset<Tensor1D<float> >& out_data, 
		      const VectorWithOffset<Tensor1D<float> >& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

template
void 
overlap_interpolate<>(VectorWithOffset<Tensor2D<float> >& out_data, 
		      const VectorWithOffset<Tensor2D<float> >& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);


#else 

#include "stir/Array.h"

START_NAMESPACE_STIR

template
void 
overlap_interpolate<>(VectorWithOffset<Array<1,float> >& out_data, 
		      const VectorWithOffset<Array<1,float> >& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

template
void 
overlap_interpolate<>(VectorWithOffset<Array<2,float> >& out_data, 
		      const VectorWithOffset<Array<2,float> >& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);
#endif

END_NAMESPACE_STIR
